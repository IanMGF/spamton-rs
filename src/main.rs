use std::ops::Div;

use nalgebra::SMatrix;
use rand::thread_rng;
use rand::{seq::SliceRandom, Rng};
use spamton_rs::email::{EmailType, Entry, EntryFeatures};
use statrs::distribution::{Continuous, Normal};

const RATIOS: (f64, f64, f64) = (0.8, 0.1, 0.1);

fn main() {
    let mut reader = csv::Reader::from_path("spambase/spambase.data").unwrap();

    let Ok(mut data): Result<Vec<Entry>, _> = reader.deserialize().collect() else {
        panic!("Failed to parse entries");
    };

    data.shuffle(&mut thread_rng());
    let data_size = data.len();

    println!("Parsed {} entries", data_size);

    let (adjust_data, validation_data, testing_data): (Vec<_>, Vec<_>, Vec<_>) = {
        let training_size = (data_size as f64 * (RATIOS.0 + RATIOS.1)).round() as usize;
        let adjust_size = (data_size as f64 * RATIOS.0).round() as usize;

        let split_data = data.split_at(training_size);
        let (split_data, testing_data) = (split_data.0.split_at(adjust_size), split_data.1.into());
        let (adjust_data, validation_data) = (split_data.0.into(), split_data.1.into());

        (adjust_data, validation_data, testing_data)
    };

    println!(
        "Hold-out distribution: ({}, {}, {})",
        adjust_data.len(),
        validation_data.len(),
        testing_data.len()
    );

    let spam_data = adjust_data
        .iter()
        .filter(|entry| entry.1 == EmailType::Spam)
        .cycle()
        .take(adjust_data.len())
        .collect::<Vec<_>>();
    let ham_data = adjust_data
        .iter()
        .filter(|entry| entry.1 == EmailType::Ham)
        .cycle()
        .take(adjust_data.len())
        .collect::<Vec<_>>();

    println!(
        "Class Distribution: (Spam: {}, Ham: {})",
        spam_data.len(),
        ham_data.len()
    );

    let spam_dist = train::<57>(&spam_data);
    let ham_dist = train::<57>(&ham_data);

    let mut confusion_matrix = SMatrix::<u64, 2, 2>::zeros();

    for entry in validation_data {
        let spam_probability = log_probability::<57>(&entry.0, &spam_dist);
        let ham_probability = log_probability::<57>(&entry.0, &ham_dist);

        let predicted_email_type = if spam_probability > ham_probability {
            EmailType::Spam
        } else if spam_probability < ham_probability {
            EmailType::Ham
        } else {
            match Rng::gen_bool(&mut rand::thread_rng(), 0.5) {
                true => EmailType::Spam,
                false => EmailType::Ham,
            }
        };

        let index = match (entry.1, predicted_email_type) {
            (EmailType::Spam, EmailType::Spam) => (0, 0),
            (EmailType::Spam, EmailType::Ham) => (0, 1),
            (EmailType::Ham, EmailType::Spam) => (1, 0),
            (EmailType::Ham, EmailType::Ham) => (1, 1),
        };

        confusion_matrix[index] += 1;
    }

    println!("Confusion matrix: {}", confusion_matrix);
    println!(
        "Error: {}",
        (confusion_matrix[(0, 1)] + confusion_matrix[(1, 0)]) as f64
            / confusion_matrix.sum() as f64
    );
}

pub fn train<const N: usize>(data: &[&Entry]) -> [Normal; N] {
    let dist = (0..N)
        .map(|i| {
            let mean = data.iter().map(|d| d.get_feature(i).unwrap()).sum::<f64>().div(data.len() as f64);
            let std_dev = data
                .iter()
                .map(|entry| (entry.get_feature(i).unwrap() - mean).powi(2))
                .sum::<f64>()
                .div(data.len() as f64)
                .sqrt();
        
            let std_dev = f64::max(std_dev, 10e-32);
            let mean = f64::max(mean, 10e-32);
        
            Normal::new(mean, std_dev).unwrap()
        })
        .collect::<Vec<_>>()
        .try_into()
        .ok();

    match dist {
        None => panic!(),
        Some(d) => d,
    }
}

pub fn log_probability<const N: usize>(
    features: &EntryFeatures,
    dist: &[Normal; N],
) -> f64 {
    // Instead of calculating the product of P(X = n), to help keep accuracy, we calculate the sum of log_2(P(x = n))
    let mut log_probability = 0.;

    (0..N).for_each(|i| {
        let feat_dist = &dist[i];
        let x = features.0[i];

        log_probability += feat_dist.pdf(x).max(1e-32).ln();
    });

    log_probability
}
