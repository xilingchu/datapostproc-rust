// src/data.rs
use std::collections::HashMap;
use hdf5::{Dataset, H5Type};
use crate::hdf5::Block;

pub struct H5File 
{
    filename: String,
    variables: Vec<String>,
    datasets: HashMap<String, Data>,
}

pub struct Data
{
    name: String,
    block: Block,
    dataset: Dataset,
}

impl H5File {
    fn new(self, filename: String) -> Self {
        Self {
            filename,
            variables: Vec::new(),
            datasets: HashMap::new(),
        }
    }

    fn add_dataset(&mut self, name: &str, data:Data) {
        self.variables.push(String::from(name));
        self.datasets.insert(String::from(name), data);
    }
}

impl Data {
    pub fn new(name:String, block: Block, dataset: Dataset) -> Self {
        Self {name, block, dataset}
    }
}
