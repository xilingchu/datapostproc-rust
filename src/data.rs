// src/data.rs
use std::collections::HashMap;
use hdf5::{Dataset, File, Error, H5Type};
use crate::hdf5::{Block, HdfOper, H5Data};

pub struct H5File 
{
    file: File,
    filename: String,
    variables: Vec<String>,
    coords : HashMap<String, Data>,
    datasets: HashMap<String, Data>,
}

pub struct Data
{
    name: String,
    block: Block,
    dataset: Dataset,
}

impl H5File {
    fn new(self, filename: &str) -> Result<Self, Error> {
        let file = File::open(&filename)?;
        Ok(Self {
            file,
            filename: String::from(filename),
            variables: Vec::new(),
            coords: HashMap::new(),
            datasets: HashMap::new(),
        })
    }

    fn add_coordinate(&mut self, name: &str, block:Block) -> Result<(), Error> {
        self.coords.push(String::from(name));
        let dataset = self.file.dataset(name)?;
        let data = Data::new(String::from(name), block, dataset);
        self.datasets.insert(String::from(name), data);
        Ok(())
    }

    fn add_dataset(&mut self, name: &str, block:Block) -> Result<(), Error> {
        self.variables.push(String::from(name));
        let dataset = self.file.dataset(name)?;
        let data = Data::new(String::from(name), block, dataset);
        self.datasets.insert(String::from(name), data);
        Ok(())
    }
}

impl Data {
    pub fn new(name:String, block: Block, dataset: Dataset) -> Self {
        Self {name, block, dataset}
    }

    // Read data through chunking
    fn read_data<T>(
            &self,
            dataset:Dataset,
            block:Block,
        ) -> Result<H5Data<T>>
        where
            T: H5Type + Copy
    {
        if dataset.is_single() {
            dataset.read_1d::<T>()?.first().copied().map(H5Data::Scalar).ok_or_else(|| Error::from("Empty dataset"))
        } else {
            dataset.read_hyperslab(block).map(H5Data::Array)
        }
    }
}

impl HdfOper for H5File {
    // Open file
    fn open_file<P: AsRef<std::path::Path>>(&self) -> hdf5::Result<File, Error> {
        File::open(&self.filename)
    }
    // Drop file
    fn close_file(file:File) -> Result<(), Error> {
        drop(file);
        Ok(())
    }
}
