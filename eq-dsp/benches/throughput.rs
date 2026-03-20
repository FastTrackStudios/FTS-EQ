//! Performance benchmarks for eq-dsp.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use eq_dsp::band::Band;
use eq_dsp::chain::EqChain;
use eq_dsp::filter_type::{FilterStructure, FilterType};
use eq_dsp::response;
use eq_dsp::test_util;
use fts_dsp::{AudioConfig, Processor};

const SAMPLE_RATE: f64 = 48000.0;
const BLOCK_SIZE: usize = 512;

fn config() -> AudioConfig {
    AudioConfig {
        sample_rate: SAMPLE_RATE,
        max_buffer_size: BLOCK_SIZE,
    }
}

fn bench_band_tick(c: &mut Criterion) {
    let mut group = c.benchmark_group("band_tick");
    let noise = test_util::white_noise(BLOCK_SIZE, 42);

    let cases = [
        ("peak_tdf2_o2", FilterType::Peak, FilterStructure::Tdf2, 2),
        ("peak_svf_o2", FilterType::Peak, FilterStructure::Svf, 2),
        (
            "lowpass_tdf2_o2",
            FilterType::Lowpass,
            FilterStructure::Tdf2,
            2,
        ),
        (
            "lowpass_tdf2_o8",
            FilterType::Lowpass,
            FilterStructure::Tdf2,
            8,
        ),
        (
            "lowpass_svf_o8",
            FilterType::Lowpass,
            FilterStructure::Svf,
            8,
        ),
        (
            "highshelf_tdf2_o4",
            FilterType::HighShelf,
            FilterStructure::Tdf2,
            4,
        ),
    ];

    for (name, ft, structure, order) in cases {
        group.throughput(Throughput::Elements(BLOCK_SIZE as u64));
        group.bench_with_input(BenchmarkId::new("samples", name), &noise, |b, input| {
            let mut band = Band::new();
            band.filter_type = ft;
            band.structure = structure;
            band.freq_hz = 1000.0;
            band.q = 0.707;
            band.gain_db = 6.0;
            band.order = order;
            band.update(config());

            b.iter(|| {
                for &s in input {
                    std::hint::black_box(band.tick(s, 0));
                }
            });
        });
    }
    group.finish();
}

fn bench_chain_process(c: &mut Criterion) {
    let mut group = c.benchmark_group("chain_process");
    let noise = test_util::white_noise(BLOCK_SIZE, 42);

    for band_count in [1, 4, 8, 16, 24] {
        group.throughput(Throughput::Elements(BLOCK_SIZE as u64));
        group.bench_with_input(
            BenchmarkId::new("stereo_samples", band_count),
            &noise,
            |b, input| {
                let mut chain = EqChain::new();
                chain.update(config());

                let freqs = [100.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0];
                for i in 0..band_count {
                    let idx = chain.add_band();
                    if let Some(band) = chain.band_mut(idx) {
                        band.filter_type = FilterType::Peak;
                        band.freq_hz = freqs[i % freqs.len()];
                        band.gain_db = 3.0;
                        band.q = 1.0;
                    }
                    chain.update_band(idx);
                }

                let mut left = input.clone();
                let mut right = input.clone();

                b.iter(|| {
                    left.copy_from_slice(input);
                    right.copy_from_slice(input);
                    chain.process(&mut left, &mut right);
                    std::hint::black_box(&left);
                });
            },
        );
    }
    group.finish();
}

fn bench_coeff_calculate(c: &mut Criterion) {
    let mut group = c.benchmark_group("coeff_calculate");

    let types = [
        ("peak", FilterType::Peak),
        ("lowpass", FilterType::Lowpass),
        ("highpass", FilterType::Highpass),
        ("low_shelf", FilterType::LowShelf),
        ("high_shelf", FilterType::HighShelf),
        ("tilt_shelf", FilterType::TiltShelf),
        ("bandpass", FilterType::Bandpass),
        ("notch", FilterType::Notch),
    ];

    for (name, ft) in types {
        group.bench_function(name, |b| {
            b.iter(|| {
                std::hint::black_box(eq_dsp::coeff::calculate(
                    ft,
                    1000.0,
                    0.707,
                    6.0,
                    SAMPLE_RATE,
                ));
            });
        });
    }
    group.finish();
}

fn bench_response_curve(c: &mut Criterion) {
    let mut group = c.benchmark_group("response_curve");

    for num_bands in [1, 4, 8] {
        for num_points in [256, 512] {
            let mut bands = Vec::new();
            let freqs = [
                250.0, 1000.0, 4000.0, 8000.0, 500.0, 2000.0, 6000.0, 12000.0,
            ];
            for i in 0..num_bands {
                let mut b = Band::new();
                b.filter_type = FilterType::Peak;
                b.freq_hz = freqs[i % freqs.len()];
                b.gain_db = 3.0;
                b.q = 1.0;
                b.update(config());
                bands.push(b);
            }

            group.bench_function(format!("{num_bands}bands_{num_points}pts"), |b| {
                b.iter(|| {
                    std::hint::black_box(response::response_curve(&bands, SAMPLE_RATE, num_points));
                });
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_band_tick,
    bench_chain_process,
    bench_coeff_calculate,
    bench_response_curve
);
criterion_main!(benches);
