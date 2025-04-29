use crate::location::Span;
use crate::types::Type;
use std::collections::HashMap;
use crate::typechecker::StructDefinition;

// Reusing VariableInfo from codegen for now, might become specific later
// We need mutability info here too.
#[derive(Debug, Clone, PartialEq)]
pub struct SymbolInfo {
    pub ty: Type,
    pub is_mutable: bool,
    // Add more later: definition location, etc.
    pub def_span: Span, // Span where variable/param was defined
}

// Represents Function Signature Info for type checking
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionSignature {
    pub param_types: Vec<Type>,
    pub return_type: Type,
    pub def_span: Span, // Maybe store reference to definition AST node?
}

#[derive(Debug, Clone, Default)]
struct Scope {
    variables: HashMap<String, SymbolInfo>,
    // Functions are often global, but could be nested later
    // functions: HashMap<String, FunctionSignature>,
}

#[derive(Debug, Default)]
pub struct SymbolTable {
    scopes: Vec<Scope>,
    // Keep functions global for now
    functions: HashMap<String, FunctionSignature>,
    pub(crate) structs: HashMap<String, StructDefinition>, // Added: Struct name -> Definition
}

impl SymbolTable {
    pub fn new() -> Self {
        // Start with a single global scope
        SymbolTable {
            scopes: vec![Scope::default()], // Global scope
            functions: HashMap::new(),
            structs: HashMap::new(), // Initialize empty struct map
        }
    }

    /// Define a struct type globally.
    pub fn define_struct(&mut self, definition: StructDefinition) -> Result<(), Span> {
        if self.structs.contains_key(&definition.name) {
            let prev_span = &self.structs.get(&definition.name).unwrap().def_span;
            Err(prev_span.clone())
        } else {
            self.structs.insert(definition.name.clone(), definition);
            Ok(())
        }
    }

    /// Look up a struct definition by name.
    pub fn lookup_struct(&self, name: &str) -> Option<&StructDefinition> {
        self.structs.get(name)
    }

    /// Enter a new lexical scope (e.g., function body, block)
    pub fn enter_scope(&mut self) {
        self.scopes.push(Scope::default());
    }

    /// Exit the current lexical scope
    pub fn exit_scope(&mut self) {
        // Should not exit the global scope
        if self.scopes.len() > 1 {
            self.scopes.pop();
        } else {
            // Consider panicking or logging error if trying to exit global scope
            // For now, just a warning
            eprintln!("Warning: Attempted to exit global scope");
        }
    }

    /// Define a variable in the *current* scope.
    /// Returns error if already defined in the current scope.
    pub fn define_variable(&mut self, name: &str, info: SymbolInfo) -> Result<(), Span> {
        let current_scope = self
            .scopes
            .last_mut()
            .expect("Symbol table scope stack empty");
        if current_scope.variables.contains_key(name) {
            let prev_span = &current_scope.variables.get(name).unwrap().def_span;
            Err(prev_span.clone())
        } else {
            current_scope.variables.insert(name.to_string(), info);
            Ok(())
        }
    }

    /// Look up a variable, searching from current scope outwards.
    pub fn lookup_variable(&self, name: &str) -> Option<&SymbolInfo> {
        for scope in self.scopes.iter().rev() {
            // Search from inner to outer
            if let Some(info) = scope.variables.get(name) {
                return Some(info);
            }
        }
        None // Not found in any scope
    }

    /// Check if a variable exists in the *current* scope only.
    pub fn is_defined_in_current_scope(&self, name: &str) -> bool {
        self.scopes
            .last()
            .is_some_and(|scope| scope.variables.contains_key(name))
    }

    /// Define a function (globally for now).
    /// Returns error if already defined.
    // todo: define the function in the current scope
    pub fn define_function(
        &mut self,
        name: &str,
        signature: FunctionSignature,
    ) -> Result<(), Span> {
        if self.functions.contains_key(name) {
            let prev_span = &self.functions.get(name).unwrap().def_span;
            Err(prev_span.clone())
        } else {
            self.functions.insert(name.to_string(), signature);
            Ok(())
        }
    }

    /// Look up a function signature.
    pub fn lookup_function(&self, name: &str) -> Option<&FunctionSignature> {
        self.functions.get(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn define_variable_in_current_scope() {
        let mut symbol_table = SymbolTable::new();
        let var_info = SymbolInfo {
            ty: Type::Int,
            is_mutable: true,
            def_span: Default::default(),
        };
        assert!(symbol_table.define_variable("x", var_info).is_ok());
        assert!(symbol_table.is_defined_in_current_scope("x"));
    }

    #[test]
    fn redefine_variable_in_same_scope() {
        let mut symbol_table = SymbolTable::new();
        let var_info = SymbolInfo {
            ty: Type::Int,
            is_mutable: true,
            def_span: Default::default(),
        };
        symbol_table.define_variable("x", var_info.clone()).unwrap();
        let result = symbol_table.define_variable("x", var_info);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), Default::default());
    }

    #[test]
    fn lookup_variable_in_outer_scope() {
        let mut symbol_table = SymbolTable::new();
        let var_info = SymbolInfo {
            ty: Type::Int,
            is_mutable: true,
            def_span: Default::default(),
        };
        symbol_table.define_variable("x", var_info.clone()).unwrap();
        symbol_table.enter_scope();
        assert_eq!(symbol_table.lookup_variable("x"), Some(&var_info));
    }

    #[test]
    fn lookup_variable_not_found() {
        let mut symbol_table = SymbolTable::new();
        assert!(symbol_table.lookup_variable("y").is_none());
    }

    #[test]
    fn exit_global_scope_warning() {
        let mut symbol_table = SymbolTable::new();
        symbol_table.exit_scope(); // Should not panic
    }

    #[test]
    fn define_function_successfully() {
        let mut symbol_table = SymbolTable::new();
        let func_sig = FunctionSignature {
            param_types: vec![Type::Int],
            return_type: Type::Void,
            def_span: Default::default(),
        };
        assert!(symbol_table
            .define_function("foo", func_sig.clone())
            .is_ok());
        assert_eq!(symbol_table.lookup_function("foo"), Some(&func_sig));
    }

    #[test]
    fn redefine_function_error() {
        let mut symbol_table = SymbolTable::new();
        let func_sig = FunctionSignature {
            param_types: vec![Type::Int],
            return_type: Type::Void,
            def_span: Default::default(),
        };
        symbol_table
            .define_function("foo", func_sig.clone())
            .unwrap();
        let result = symbol_table.define_function("foo", func_sig);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), Default::default());
    }
}
