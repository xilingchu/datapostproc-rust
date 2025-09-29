// src/hdf5.rs
// High-level hdf5 method using hdf5-rust

use hdf5::{File, Dataset, Error, Result, Selection, types::{TypeDescriptor, FloatSize, IntSize}};
use std::path::Path;
use ndarray::ArrayD;

// Macros
// For hdf5.rs
// Build the array to carry the value.
macro_rules! array_create {
    ($name:ident, $data_shape:expr, $data_type:ty) => {
        #[allow(used_mut)]
        let mut $name = match $data_shape.as_slice() {
            &[1] => {
                TypeData::Scalar(<$data_type>::default())
            }
            _ => {
                TypeData::Array(ArrayD::<$data_type>::zeros($data_shape))
            }
        };
    }
}

#[derive(Debug)]
pub enum TypeData<T>{
    Scalar(T),
    Array(ArrayD<T>)
}

// Meaning of block
// Details of blockx(yz) are blockx[start, stride, count, block]
// start  ---- The start point of the block
// stride ---- The length of two blocks
// count  ---- The count of the blocks
// block  ---- The size of one block
#[derive(Debug, Clone, Copy)]
pub struct BlockValue([usize; 4]);

impl BlockValue {
    fn new(values: [usize; 4]) -> Result<Self, String> {
        if values[3] >= values[1] {
            Err("Stride should be higher than block.".into())
        } else {
            Ok(Self(values))
        }
    }
    
    fn lb(&self) -> usize {
        self.0[0]
    }

    fn rb(&self) -> usize {
        self.0[0]+self.0[3]*(self.0[2]+self.0[4]-1)
    }
}



#[derive(Debug, Clone)]
pub struct Block(Vec<Option<BlockValue>>);

impl Block {
    // Validate block bounds
    fn validate_bounds(&self, shape: &[usize]) -> Result<Self> {
        let len = shape.len();
        let block = &self.0.clone()[0..len];
        // let dim = ['z', 'y', 'x'];
        fn check_bound(blocki: Option<BlockValue>, b:usize) -> Result<Option<BlockValue>> {
            if blocki.is_some() {
                if blocki.unwrap().rb() < b {
                    return Err(format!("Block exceeds bounds in dimension.").into());
                } else {
                    Ok(blocki)
                }
            } else {
                Ok(Some(BlockValue::new([1, 1, b, 1])?))
            }
        }

        // Renew the bounds
        let mut block_out = Vec::new();
        let mut i = 0usize;
        for block_item in block {  
            i += 1;
            block_out.push(check_bound(*block_item, shape[i])?);
        }

        Ok(Block(block_out))
    }

    fn dim(&self) -> usize {
        let mut count:usize = 0;
        for block_item in &self.0{
            if block_item.is_some() {
                count += 1
            }
        }
        count
    }

    fn size(&self) -> Vec<usize> {
        let mut size = Vec::new();
        for block_item in &self.0{
            if let Some(bx) = &block_item {
                size.push(bx.0[2] * bx.0[3]);
            }
        }
        size
    }

    fn build_hyberslab_selection(&self) -> Result<Selection> {

    }
}

pub trait HdfOper{
    fn open_file<P: AsRef<Path>>(filename: P) -> Result<File, Error> {
        File::open(filename)
    }

    fn close_file(file:File) -> Result<(), Error> {
        drop(file);
        Ok(())
    }

    // Read data through chunking
    fn chunking(dataset:Dataset, block:Block, shape:&[usize]) -> Result<ArrayD<_>, Error> {
        let size = block.size();
        let mut read_array = ArrayD::<_>::zeros(size);
        OK(read_array)
    }

    // Read data
    fn read_data(
        file:File,
        dataset:&str,
        block:Block,
        shape_bounds:&[usize]
        ) -> Result<Hdf5Data, Error> {
        let dataset = file.dataset(dataset)?;
        ////////Validation of the dataset and block////////
        // Get the type of dataset
        let dtype = dataset.dtype()?;
        // Get the chunk through chunkx, chunky, and chunkz.
        // Step 1. Check the shape is legal or not.
        let shape = dataset.shape();
        let len = shape.len();
        if block.dim() > len.try_into().unwrap(){
            Error::from("The shape of block has error in dimension!");
        }
        // Step 2. Control the region to read.
        block.validate_bounds(&shape);
        // match dtype
        match dtype.to_descriptor()? {
            TypeDescriptor::Integer(size) => match size {
                IntSize::U8 => {
                    if dataset.shape() == [1] {  // Scalar
                        dataset.read_1d::<i64>()?.first().copied().map(Hdf5Data::I64).ok_or_else(|| Error::from("Empty dataset"))
                    } else {
                        dataset.read_dyn::<i64>().map(Hdf5Data::ArrayI64);
                        let mut input = ArrayD::<i64>::zeros(shape_bounds);
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
