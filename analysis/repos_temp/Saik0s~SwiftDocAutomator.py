"""
You generate documentation comments for provided Swift functions, following the official Apple and Swift guidelines. The comment include:

1. A concise description of the function's purpose and data flow.
2. A list of the function's parameters, with a description for each.
3. A description of the function's return value, if applicable.
4. Any additional notes or context, if necessary.

Example function:
internal static func _typeMismatch(at path: [CodingKey], expectation: Any.Type, reality: Any) -> DecodingError {
    let description = "Expected to decode \(expectation) but found \(_typeDescription(of: reality)) instead."
    return .typeMismatch(expectation, Context(codingPath: path, debugDescription: description))
}

Generated comment:
/// Returns a `.typeMismatch` error describing the expected type.
///
/// - parameter path: The path of `CodingKey`s taken to decode a value of this type.
/// - parameter expectation: The type expected to be encountered.
/// - parameter reality: The value that was encountered instead of the expected type.
/// - returns: A `DecodingError` with the appropriate path and debug description.
""""""
Function implementation:
```
{function_implementation}
```

Please provide the documentation comment based on the given function implementation.
""""""Write a concise standalone documentation comment for a type described by code or comments, following the official Apple and Swift guidelines:

"{text}"

documentation comment where every line starts with ///:""""""
    @usableFromInline
    func typeName(_ type: Any.Type) -> String {
    var name = _typeName(type, qualified: true)
    if let index = name.firstIndex(of: ".") {
        name.removeSubrange(...index)
    }
    let sanitizedName =
        name
        .replacingOccurrences(
        of: #"<.+>|\(unknown context at \$[[:xdigit:]]+\)\."#,
        with: "",
        options: .regularExpression
        )
    return sanitizedName
    }
    """