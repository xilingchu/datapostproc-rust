use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Data, Fields};

/// Derive macro that adds an `iter_fields` method to a struct.
///
/// The generated method returns an iterator of `(&'static str, &dyn std::any::Any)` tuples,
/// where the first element is the field name and the second is a reference to the field value.
///
/// # Requirements
/// - Only works on structs with named fields.
/// - All field types must be `'static` (i.e., contain no non-static references).
///
/// # Example
/// ```rust
/// use macro_struct::IterFields;
///
/// #[derive(IterFields)]
/// struct Point { x: f64, y: f64 }
///
/// let p = Point { x: 1.0, y: 2.0 };
/// for (name, val) in p.iter_fields() {
///     println!("{name} = {:?}", val.downcast_ref::<f64>());
/// }
/// ```
#[proc_macro_derive(IterFields)]
pub fn iter_fields_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => panic!("IterFields only supports structs with named fields"),
        },
        _ => panic!("IterFields only supports structs"),
    };

    let field_entries: Vec<TokenStream2> = fields
        .iter()
        .map(|f| {
            let field_name = f.ident.as_ref().unwrap();
            let field_name_str = field_name.to_string();
            quote! {
                (#field_name_str, &self.#field_name as &dyn std::any::Any)
            }
        })
        .collect();

    let expanded = quote! {
        impl #name {
            pub fn iter_fields(&self) -> impl Iterator<Item = (&'static str, &dyn std::any::Any)> + '_ {
                let v: Vec<(&'static str, &dyn std::any::Any)> = vec![
                    #(#field_entries),*
                ];
                v.into_iter()
            }
        }
    };

    expanded.into()
}
