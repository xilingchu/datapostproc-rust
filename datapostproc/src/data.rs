// src/data.rs
use std::collections::HashMap;
use hdf5::{Dataset, File, Error, H5Type};
use crate::hdf5::{Block, H5Data, DatasetHyperslabExt};


pub struct H5File
{
    file: File,
    filename: String,
    info: DNSInfo,
    variables: Vec<String>,
    coords : HashMap<String, Data>,
    datasets: HashMap<String, Data>,
}

pub struct Data
{
    name: String,
    block: Option<Block>,
    dataset: Dataset,
}

#[derive(Debug, Default)]
pub struct DNSInfo {
    nx: Option<i32>,
    ny: Option<i32>,
    nz: Option<i32>,
    lx: Option<f64>,
    ly: Option<f64>,
    lz: Option<f64>,
    re: Option<f64>,
    dt: Option<f64>,
    is_periodic: bool,
    is_defined: bool
}

impl H5File {
    fn new(filename: &str) -> Result<Self, Error> {
        let file = File::open(&filename)?;
        Ok(Self {
            file,
            filename: String::from(filename),
            info: DNSInfo::default(),
            variables: Vec::new(),
            coords: HashMap::new(),
            datasets: HashMap::new(),
        })
    }

        let dataset = self.file.dataset(name)?;
        let data = Data::new(String::from(name), block, dataset);
        self.coords.insert(String::from(name), data);
        Ok(())
    }

    fn add_dataset(&mut self, name: &str, block:Option<Block>) -> Result<(), Error> {
        self.variables.push(String::from(name));
        let dataset = self.file.dataset(name)?;
        let data = Data::new(String::from(name), block, dataset);
        self.datasets.insert(String::from(name), data);
        Ok(())
    }
    
    fn get_info(&mut self) -> Result<(), Error> {
        info_iter = 
    }
}

impl Data{
    pub fn new(name:String, block: Option<Block>, dataset: Dataset) -> Self {
        Self {name, block, dataset}
    }

    // Read data through chunking
    fn read_data<T>(
            &self,
        ) -> Result<H5Data<T>, Error>
        where
            T: H5Type + Copy
    {
        match &self.block {
            None => {
                if self.dataset.is_single() {
                    self.dataset.read_1d::<T>()?.first().copied().map(H5Data::Scalar).ok_or_else(|| Error::from("Empty dataset"))
                } else {
                    self.dataset.read_dyn().map(H5Data::Array)
                }
            }
            Some(block) => {
                if self.dataset.is_single() {
                    self.dataset.read_1d::<T>()?.first().copied().map(H5Data::Scalar).ok_or_else(|| Error::from("Empty dataset"))
                } else {
                    self.dataset.read_hyperslab(block.clone()).map(H5Data::Array)
                }
            }
        }
    }
}
