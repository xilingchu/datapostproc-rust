use std::collections::HashMap;
// src/data.rs
use ndarray::{ArrayD, Array1};

pub struct H5file<'a, T> 
where
    T: ndarray::NdFloat
{
    filename: &'a str,
    data: HashMap<String, Data<'a, T>>,
}

pub struct Data<'a, T> 
where 
    T: ndarray::NdFloat
{
    name: &'a str,
    dimension: Array1<i32>,
    dataset: ArrayD<T>,
}
