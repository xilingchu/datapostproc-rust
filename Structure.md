# Structure of datapostproc
## The basic structure
### File
- Name: str
- Loc : str
- DNSInfo : struct
- Variables:
    - name: str
    - type: type
    - field: Option<Array<A,D>>

### Data
- Name: str 
- Type: type
- Field: Array<A,D>

### DNSInfo
- nx: Option<i32>
- ny: Option<i32>
- nz: Option<i32>
- lx: Option<f64>
- ly: Option<f64>
- lz: Option<f64>
- re: Option<f64>
- periodic: bool
- defined:  bool

### Tools
