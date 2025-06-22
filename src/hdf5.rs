// src/hdf5.rs
// High-level hdf5 method using hdf5-rust

use hdf5::{File, Error, Result, types::{TypeDescriptor, FloatSize, IntSize}};
use std::path::Path;
use ndarray::ArrayD;

#[derive(Debug)]
pub enum Hdf5Data {
    ArrayF64(ArrayD<f64>),
    ArrayF32(ArrayD<f32>),
    ArrayI64(ArrayD<i64>),
    ArrayI32(ArrayD<i32>),
    I64(i64),
    I32(i32),
    F64(f64),
    F32(f32)
}

pub trait HdfOper{
    fn open_file<P: AsRef<Path>>(filename: P) -> Result<File, Error> {
        File::open(filename)
    }

    fn close_file(file:File) -> Result<(), Error> {
        drop(file);
        Ok(())
    }

    fn read_data<T>(file:File, dataset:&str) -> Result<Hdf5Data, Error> {
        let dataset = file.dataset(dataset)?;
        // Get the type of dataset
        let dtype = dataset.dtype()?;
        // match dtype
        match dtype.to_descriptor()? {
            TypeDescriptor::Integer(size) => match size {
                IntSize::U8 => {
                    if dataset.shape() == [1] {  // Scalar
                        dataset.read_1d::<i64>()?.first().copied().map(Hdf5Data::I64).ok_or_else(|| Error::from("Empty dataset"))
                    } else {
                        dataset.read_dyn::<i64>().map(Hdf5Data::ArrayI64)
                    }
                },
                IntSize::U4 => {
                    if dataset.shape() == [1] {
                        dataset.read_1d::<i32>()?.first().copied().map(Hdf5Data::I32).ok_or_else(|| Error::from("Empty dataset"))
                    } else {
                        dataset.read_dyn::<i32>().map(Hdf5Data::ArrayI32)
                    }
                },
                _ => Err(hdf5::Error::from("Unsupported integer size")),
            },
            TypeDescriptor::Float(size) => match size {
                FloatSize::U8 => {
                    if dataset.shape() == [1] {  // Scalar
                        dataset.read_1d::<f64>()?.first().copied().map(Hdf5Data::F64).ok_or_else(|| Error::from("Empty dataset"))
                    } else {
                        dataset.read_dyn::<f64>().map(Hdf5Data::ArrayF64)
                    }
                },
                FloatSize::U4 => {
                    if dataset.shape() == [1] {
                        dataset.read_1d::<f32>()?.first().copied().map(Hdf5Data::F32).ok_or_else(|| Error::from("Empty dataset"))
                    } else {
                        dataset.read_dyn::<f32>().map(Hdf5Data::ArrayF32)
                    }
                },
                #[allow(unreachable_patterns)]
                _ => Err(hdf5::Error::from("Unsupported integer size")),
            },
            _ => Err(hdf5::Error::from("Unsupported type"))
        }
    }
}
