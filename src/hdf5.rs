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

// Meaning of block
// Details of blockx(yz) are blockx[start, stride, count, block]
// start  ---- The start point of the block
// stride ---- The length of two blocks
// count  ---- The count of the blocks
// block  ---- The size of one block
#[derive(Debug, Clone)]
pub struct BlockValue([i32; 4]);

impl BlockValue {
    pub fn new(values: [i32; 4]) -> Result<Self, String> {
        if values[1] <= 0 || values[2] <= 0 || values[3] <= 0{
            Err("Stride, count and block must be positive".into())
        } else {
            Ok(Self(values))
        }
    }
}

pub struct Block {
    blockx: Option<BlockValue>,
    blocky: Option<BlockValue>,
    blockz: Option<BlockValue>
}

impl Block {
    fn validate_bounds(&self, shape: &[usize]) -> Result<Self, String> {
        let rb = self.rb();
        let len = shape.len();
        let mut blockx = self.blockx.clone();
        let mut blocky = self.blocky.clone();
        let mut blockz = self.blockz.clone();
        // z
        if len >= 1 {
            if let Some(rbz) = rb[2] {
                if rbz > shape[2] as i32 {
                    return Err("Block exceeds x dimension bounds".into());
                } else {
                    if self.blockz.is_none() {
                        blockz = Some(BlockValue::new([1, 1, shape[2].try_into().unwrap(), 1])?);
                    }
                }
            }
            // y
            if len >= 2 {
                if let Some(rby) = rb[1] {
                    if rby > shape[2] as i32 {
                        return Err("Block exceeds y dimension bounds".into());
                    } else {
                        if self.blocky.is_none() {
                            blocky = Some(BlockValue::new([1, 1, shape[1].try_into().unwrap(), 1])?);
                        }
                    }
                }
                if len >= 3 {
                    if let Some(rbx) = rb[0] {
                        if rbx > shape[0] as i32 {
                            return Err("Block exceeds x dimension bounds".into());
                        } else {
                            if self.blockx.is_none() {
                                blockx = Some(BlockValue::new([1, 1, shape[1].try_into().unwrap(), 1])?);
                            }
                        }
                    }
                }
            }
        }
        Ok(Self { blockx: blockx, blocky: blocky, blockz:blockz })
    }

    fn dim(&self) -> i32 {
        let mut count:i32 = 0;
        if self.blockx.is_some() {
            count += 1;
        }
        if self.blocky.is_some() {
            count += 1;
        }
        if self.blockz.is_some() {
            count += 1;
        }
        count
    }
    
    fn rb(&self) -> [Option<i32>; 3] {
        let mut result = [None, None, None];
        if let Some(BlockValue(bx)) = self.blockx {
            result[0] = Some(bx[0]+bx[3]*(bx[2]+bx[4]-1));
        }
        if let Some(BlockValue(by)) = self.blocky {
            result[1] = Some(by[0]+by[3]*(by[2]+by[4]-1));
        }
        if let Some(BlockValue(bz)) = self.blockz {
            result[2] = Some(bz[0]+bz[3]*(bz[2]+bz[4]-1));
        }
        result
    }

    fn lb(&self) -> [Option<i32>; 3] {
        let mut result = [None, None, None];
        if let Some(BlockValue(bx)) = self.blockx {
            result[0] = Some(bx[0]);
        }
        if let Some(BlockValue(by)) = self.blocky {
            result[1] = Some(by[0]);
        }
        if let Some(BlockValue(bz)) = self.blockz {
            result[2] = Some(bz[0]);
        }
        result
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

    fn read_data(
        file:File, 
        dataset:&str,
        block: Block,
        ) -> Result<Hdf5Data, Error> {
        let dataset = file.dataset(dataset)?;
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
