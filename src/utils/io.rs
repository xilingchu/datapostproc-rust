use hdf5::filters::blosc_set_nthreads;
use hdf5::{File, Result, Error, types::{TypeDescriptor, FloatSize, IntSize}};
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

pub fn read_hdf5_file(filename: &str, data: &str) -> Result<Hdf5Data> {
    // Open HDF5 file
    let file = File::open(filename)?;
    // Read a dataset
    blosc_set_nthreads(2);
    let dataset = file.dataset(data)?;
    // Get the datatype
    let dtype = dataset.dtype()?;
    
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
