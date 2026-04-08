use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Data, Fields};

/// Derive macro that adds `iter_fields` and `iter_fields_mut` methods to a struct.
///
/// - `iter_fields(&self)` → `Iterator<Item = (&'static str, &dyn Any)>`
/// - `iter_fields_mut(&mut self)` → `Iterator<Item = (&'static str, &mut dyn Any)>`
///
/// # Requirements
/// - Only works on structs with named fields.
/// - All field types must be `'static`.
///
/// # Example
/// ```rust
/// use macro_struct::IterFields;
///
/// #[derive(IterFields)]
/// struct Point { x: f64, y: f64 }
///
/// let mut p = Point { x: 1.0, y: 2.0 };
/// for (name, val) in p.iter_fields_mut() {
///     if let Some(v) = val.downcast_mut::<f64>() {
///         *v *= 2.0;
///     }
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

    let field_entries_mut: Vec<TokenStream2> = fields
        .iter()
        .map(|f| {
            let field_name = f.ident.as_ref().unwrap();
            let field_name_str = field_name.to_string();
            quote! {
                (#field_name_str, &mut self.#field_name as &mut dyn std::any::Any)
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

            pub fn iter_fields_mut(&mut self) -> impl Iterator<Item = (&'static str, &mut dyn std::any::Any)> + '_ {
                let v: Vec<(&'static str, &mut dyn std::any::Any)> = vec![
                    #(#field_entries_mut),*
                ];
                v.into_iter()
            }
        }
    };

    expanded.into()
}
