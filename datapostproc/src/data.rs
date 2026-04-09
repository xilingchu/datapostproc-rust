// src/data.rs
use std::collections::HashMap;
use hdf5::{Dataset, File, Error, H5Type};
use crate::hdf5::{Block, H5Data, DatasetHyperslabExt};

pub struct H5File {
    file: File,
    filename: String,
    info: DNSInfo,
    variables: Vec<String>,
    coords: HashMap<String, Data>,
    datasets: HashMap<String, Data>,
}

pub struct Data {
    name: String,
    block: Option<Block>,
    dataset: Dataset,
}

#[derive(Debug, Default)]
pub struct DNSInfo {
    pub nx: Option<i32>,
    pub ny: Option<i32>,
    pub nz: Option<i32>,
    pub re: Option<f64>,
    pub nu: Option<f64>,
    pub is_periodic: bool,
    pub is_defined: bool,
}

impl DNSInfo {
    pub fn from_h5file(file: &hdf5::File) -> hdf5::Result<Self> {
        let nx = file.dataset("x")?.shape()[0] as i32;
        let ny = file.dataset("y")?.shape()[0] as i32;
        let nz = file.dataset("zc")?.shape()[0] as i32;
        let nu = file.dataset("nu")?.read_scalar::<f64>()?;
        Ok(Self {
            nx: Some(nx),
            ny: Some(ny),
            nz: Some(nz),
            nu: Some(nu),
            re: Some(1.0 / nu),
            is_periodic: false,
            is_defined: true,
        })
    }
}

impl H5File {
    pub fn new(filename: &str) -> Result<Self, Error> {
        let file = File::open(filename)?;
        Ok(Self {
            file,
            filename: String::from(filename),
            info: DNSInfo::default(),
            variables: Vec::new(),
            coords: HashMap::new(),
            datasets: HashMap::new(),
        })
    }

    pub fn add_dataset(&mut self, name: &str, block: Option<Block>) -> Result<(), Error> {
        self.variables.push(String::from(name));
        let dataset = self.file.dataset(name)?;
        let data = Data::new(String::from(name), block, dataset);
        self.datasets.insert(String::from(name), data);
        Ok(())
    }

    pub fn get_info(&mut self) -> Result<(), Error> {
        self.info = DNSInfo::from_h5file(&self.file)?;
        Ok(())
    }

    pub fn load_coords(&mut self) -> Result<(), Error> {
        let names = self.file.member_names()?;
        for name in names {
            if name.starts_with('x') || name.starts_with('y') || name.starts_with('z') {
                let dataset = self.file.dataset(&name)?;
                let data = Data::new(name.clone(), None, dataset);
                self.coords.insert(name, data);
            }
        }
        Ok(())
    }

    pub fn coord(&self, name: &str) -> Option<&Data> {
        self.coords.get(name)
    }

    pub fn info(&self) -> &DNSInfo {
        &self.info
    }

    pub fn info_mut(&mut self) -> &mut DNSInfo {
        &mut self.info
    }
}

impl Data {
    pub fn new(name: String, block: Option<Block>, dataset: Dataset) -> Self {
        Self { name, block, dataset }
    }

    pub fn read_data<T>(&self) -> Result<H5Data<T>, Error>
    where
        T: H5Type + Copy,
    {
        match &self.block {
            None => {
                if self.dataset.is_single() {
                    self.dataset
                        .read_1d::<T>()?
                        .first()
                        .copied()
                        .map(H5Data::Scalar)
                        .ok_or_else(|| Error::from("Empty dataset"))
                } else {
                    self.dataset.read_dyn().map(H5Data::Array)
                }
            }
            Some(block) => self.dataset.read_hyperslab(block.clone()).map(H5Data::Array),
        }
    }
}
