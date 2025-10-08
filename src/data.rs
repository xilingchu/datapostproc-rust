// src/data.rs
use std::collections::HashMap;
use hdf5::{Dataset, H5Type};
use crate::hdf5::Block;

pub struct H5file<'a, T> 
where
    T: H5Type
{
    filename: &'a str,
    var: Vec<&'a str>,
    data: HashMap<&'a str, Data<'a, T>>,
}

pub struct Data<'a, T> 
where 
    T: H5Type
{
    name: &'a str,
    block: Block,
    dataset: Dataset,
}
