// src/data.rs
use std::collections::HashMap;
use hdf5::{Dataset, H5Type};
use crate::hdf5::Block;

pub struct H5File 
{
    filename: String,
    var: Vec<String>,
    data: HashMap<String, Data>,
}

pub struct Data
{
    name: String,
    block: Block,
    dataset: Dataset,
}
