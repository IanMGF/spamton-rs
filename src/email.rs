use nalgebra::SVector;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmailType {
    Spam,
    Ham,
}

#[derive(Debug, Clone)]
pub struct EntryFeatures(pub SVector<f64, 57>);

#[derive(Debug, Clone)]
pub struct Entry(pub EntryFeatures, pub EmailType);

impl<'de> Deserialize<'de> for Entry {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Deserialize as a vector of strings first
        let vec: Vec<String> = Deserialize::deserialize(deserializer)?;

        if vec.len() != 58 {
            return Err(serde::de::Error::custom(format!(
                "Expected 58 elements, found {}",
                vec.len()
            )));
        }

        // Convert the first 57 elements to f64
        let mut array = [0.0; 57];
        for (i, elem) in vec.iter().take(57).enumerate() {
            array[i] = elem.parse::<f64>().map_err(serde::de::Error::custom)?;
        }

        // Parse the last element to a bool
        let last_elem = match vec[57].as_str() {
            "0" => EmailType::Ham,
            "1" => EmailType::Spam,
            _ => return Err(serde::de::Error::custom("Expected 0 or 1 for bool")),
        };

        Ok(Entry(EntryFeatures(array.into()), last_elem))
    }
}

impl From<Entry> for (SVector<f64, 57>, EmailType) {
    fn from(value: Entry) -> Self {
        (value.0.0, value.1)
    }
}

impl Entry {
    pub fn get_feature(&self, idx: usize) -> Option<f64> {
        self.0.0.get(idx).cloned()
    }
}