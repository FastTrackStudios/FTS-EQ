//! Process a WAV file through the EQ and write the result.
//!
//! Usage:
//!   cargo run -p eq-dsp --example process_wav -- input.wav output.wav [options]
//!
//! Options:
//!   --type <peak|lowpass|highpass|lowshelf|highshelf|tiltshelf|bandpass|notch>
//!   --freq <hz>        Center/corner frequency (default: 1000)
//!   --gain <db>        Gain in dB (default: 6)
//!   --q <value>        Q factor (default: 0.707)
//!   --order <n>        Filter order 1-12 (default: 2)
//!   --structure <tdf2|svf>  Filter structure (default: tdf2)
//!
//! Also prints performance metrics after processing.

use std::env;
use std::time::Instant;

use eq_dsp::band::Band;
use eq_dsp::filter_type::{FilterStructure, FilterType};
use fts_dsp::AudioConfig;

fn parse_filter_type(s: &str) -> FilterType {
    match s.to_lowercase().as_str() {
        "peak" => FilterType::Peak,
        "lowpass" | "lp" => FilterType::Lowpass,
        "highpass" | "hp" => FilterType::Highpass,
        "lowshelf" | "ls" => FilterType::LowShelf,
        "highshelf" | "hs" => FilterType::HighShelf,
        "tiltshelf" | "tilt" => FilterType::TiltShelf,
        "bandpass" | "bp" => FilterType::Bandpass,
        "notch" => FilterType::Notch,
        _ => {
            eprintln!("Unknown filter type: {s}");
            std::process::exit(1);
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: process_wav <input.wav> <output.wav> [options]");
        eprintln!("  --type <peak|lowpass|highpass|lowshelf|highshelf|tiltshelf|bandpass|notch>");
        eprintln!("  --freq <hz>  --gain <db>  --q <value>  --order <n>  --structure <tdf2|svf>");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];

    // Parse options
    let mut filter_type = FilterType::Peak;
    let mut freq_hz = 1000.0;
    let mut gain_db = 6.0;
    let mut q = 0.707;
    let mut order: usize = 2;
    let mut structure = FilterStructure::Tdf2;

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--type" => {
                filter_type = parse_filter_type(&args[i + 1]);
                i += 2;
            }
            "--freq" => {
                freq_hz = args[i + 1].parse().expect("Invalid frequency");
                i += 2;
            }
            "--gain" => {
                gain_db = args[i + 1].parse().expect("Invalid gain");
                i += 2;
            }
            "--q" => {
                q = args[i + 1].parse().expect("Invalid Q");
                i += 2;
            }
            "--order" => {
                order = args[i + 1].parse().expect("Invalid order");
                i += 2;
            }
            "--structure" => {
                structure = match args[i + 1].to_lowercase().as_str() {
                    "tdf2" => FilterStructure::Tdf2,
                    "svf" => FilterStructure::Svf,
                    s => {
                        eprintln!("Unknown structure: {s}");
                        std::process::exit(1);
                    }
                };
                i += 2;
            }
            other => {
                eprintln!("Unknown option: {other}");
                std::process::exit(1);
            }
        }
    }

    // Read input WAV
    let reader = hound::WavReader::open(input_path).expect("Failed to open input WAV");
    let spec = reader.spec();
    let sample_rate = spec.sample_rate as f64;
    let channels = spec.channels as usize;

    println!("Input: {input_path}");
    println!("  Sample rate: {sample_rate} Hz");
    println!("  Channels: {channels}");
    println!("  Bits per sample: {}", spec.bits_per_sample);

    // Read all samples as f64
    let samples: Vec<f64> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .into_samples::<i32>()
            .map(|s| {
                let s = s.unwrap();
                s as f64 / (1 << (spec.bits_per_sample - 1)) as f64
            })
            .collect(),
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.unwrap() as f64)
            .collect(),
    };

    let num_frames = samples.len() / channels;
    println!("  Frames: {num_frames}");
    println!("  Duration: {:.2}s", num_frames as f64 / sample_rate);

    // Configure filter
    println!("\nFilter: {filter_type:?}");
    println!("  Frequency: {freq_hz} Hz");
    println!("  Gain: {gain_db} dB");
    println!("  Q: {q}");
    println!("  Order: {order}");
    println!("  Structure: {structure:?}");

    let mut band = Band::new();
    band.filter_type = filter_type;
    band.structure = structure;
    band.freq_hz = freq_hz;
    band.gain_db = gain_db;
    band.q = q;
    band.order = order;
    band.enabled = true;
    band.update(AudioConfig {
        sample_rate,
        max_buffer_size: 512,
    });

    // Process
    let start = Instant::now();
    let mut output = samples.clone();

    for frame in 0..num_frames {
        let base = frame * channels;
        for ch in 0..channels.min(2) {
            output[base + ch] = band.tick(output[base + ch], ch);
        }
    }

    let elapsed = start.elapsed();
    let total_samples = num_frames * channels;

    println!("\nPerformance:");
    println!("  Processing time: {:.3}ms", elapsed.as_secs_f64() * 1000.0);
    println!(
        "  Throughput: {:.1} MSamples/s",
        total_samples as f64 / elapsed.as_secs_f64() / 1e6
    );
    println!(
        "  Real-time factor: {:.1}x",
        (num_frames as f64 / sample_rate) / elapsed.as_secs_f64()
    );
    println!(
        "  Per-sample: {:.1} ns",
        elapsed.as_nanos() as f64 / total_samples as f64
    );

    // Write output WAV (always 32-bit float)
    let out_spec = hound::WavSpec {
        channels: spec.channels,
        sample_rate: spec.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer =
        hound::WavWriter::create(output_path, out_spec).expect("Failed to create output WAV");
    for &s in &output {
        writer.write_sample(s as f32).unwrap();
    }
    writer.finalize().unwrap();

    println!("\nOutput: {output_path}");

    // Compute some stats
    let peak_in: f64 = samples.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
    let peak_out: f64 = output.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
    let rms_in: f64 = (samples.iter().map(|s| s * s).sum::<f64>() / samples.len() as f64).sqrt();
    let rms_out: f64 = (output.iter().map(|s| s * s).sum::<f64>() / output.len() as f64).sqrt();

    println!("\nSignal stats:");
    println!(
        "  Peak: {:.4} ({:.1} dBFS) → {:.4} ({:.1} dBFS)",
        peak_in,
        20.0 * peak_in.max(1e-30).log10(),
        peak_out,
        20.0 * peak_out.max(1e-30).log10()
    );
    println!(
        "  RMS:  {:.4} ({:.1} dBFS) → {:.4} ({:.1} dBFS)",
        rms_in,
        20.0 * rms_in.max(1e-30).log10(),
        rms_out,
        20.0 * rms_out.max(1e-30).log10()
    );
}
