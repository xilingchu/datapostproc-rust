// src/hdf5.rs
// High-level hdf5 method using hdf5-rust

use hdf5::{File, Dataset, Error, Result, Selection, Hyperslab, SliceOrIndex, H5Type, types::{TypeDescriptor, FloatSize, IntSize}};
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
pub enum H5Data<T>{
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
    fn new(values: [usize; 4]) -> Result<Self> {
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
pub struct Block{
    data: Vec<Option<BlockValue>>,
    validated: bool,
}

impl Block {
    pub fn new(values: Vec<Option<BlockValue>>) -> Self {
        Self{
            data: values,
            validated: false,
        }
    }
    // Validate block bounds
    fn validate_bounds(&mut self, shape: &[usize]) -> Result<()> {
        if self.validated {
            return Ok(());
        }
        // Check bound function
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
        // Get the block
        let len = shape.len();
        let block = &self.data.clone()[0..len];
        if self.data.len() != len {
            return Err(format!("The length of hyperslab is not match the length of the dataset.").into())
        }
        else {
             // Renew the bounds
            let mut block_out = Vec::new();
            let mut i = 0usize;
            for block_item in block {  
                i += 1;
                block_out.push(check_bound(*block_item, shape[i])?);
            }
            self.data = block_out;
            self.validated = true;
            Ok(())
        }
    }

    fn dim(&self) -> Result<usize> {
        if !self.validated {
            return Err("Block must be validated before use".into());
        }
        Ok(self.data.iter().count())
    }

    fn size(&self) -> Result<Vec<usize>> {
        if !self.validated {
            return Err("Block must be validated before use".into());
        }

        Ok(self.data
            .iter()
            .filter_map(|item| item.as_ref())
            .map(|bx| bx.0[2] * bx.0[3])
            .collect())
    }

    fn build_hyberslab_selection(&self) -> Result<Selection> {
        if !self.validated {
            return Err("Block must be validated before use".into());
        }

        let ndim = self.dim()?;
        let mut slices = Vec::with_capacity(ndim);
        for i in 0..ndim {
            let bv = &self.data[i].as_ref().ok_or("Missing block value in dimension.")?;
            let slice = SliceOrIndex::SliceCount{
                start: bv.0[0],
                step: bv.0[1],
                count: bv.0[2],
                block: bv.0[3],
            };
            slices.push(slice);
        }
        let hyperslab = Hyperslab::from(slices);
        Ok(Selection::from(hyperslab))
    }
}

pub trait DatasetHyperslabExt {
    /// if single?
    fn is_single(&self) -> bool;
    /// Read from hyperslab
    fn read_hyperslab<T>(
        &self,
        block:Block,
    ) -> Result<ArrayD<T>> 
    where
        T: H5Type;
    /// Write from hyperslab
    fn write_hyperslab<T>(
        &self,
        data: &ArrayD<T>,
        block:Block,
    ) -> Result<()> 
    where
        T: H5Type;
}

impl DatasetHyperslabExt for Dataset {
    fn is_single(&self) -> bool {
        self.shape() == [1]
    }

    /// Read through hyperslab
    fn read_hyperslab<T>(
        &self,
        mut block:Block,
    ) -> Result<ArrayD<T>>
    where
        T: H5Type {
        if self.is_single() {
            return Err("Hyperslab should not be used in single data.".into());
        }
        let shape = &self.shape();
        block.validate_bounds(&shape)?;
        let select = block.build_hyberslab_selection()?;
        self.read_slice(select)
    }

    /// Write from hyperslab
    fn write_hyperslab<T>(
        &self,
        data: &ArrayD<T>,
        mut block:Block,
    ) -> Result<()> 
    where
        T: H5Type {
        if self.is_single() {
            return Err("Hyperslab should not be used in single data.".into());
        }
        // Get the shape of the block
        let shape = &self.shape();
        block.validate_bounds(&shape)?;
        let select = block.build_hyberslab_selection()?;
        // Get the shape of the data
        let size = block.size()?;
        if data.shape() != size {
            return Err(format!("Data shape {:?} does not match expected hyperslab shape {:?}", 



                    data.shape(), size).into());
        }

        self.write_slice(data, select)
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
    fn read_data(
            dataset:Dataset,
            mut block:Block,
            shape:&[usize]
        ) -> Result<H5Data<T>>
        where
            T: H5Type
    {
        let size = block.size();
        if dataset.is_single() {
            dataset.read_1d::<T>()?.first().copied().map(Hdf5Data::Scalar).ok_or_else(|| Error::from("Empty dataset"))

        }
        
    }

    // Read data
    fn rsead_data(
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
