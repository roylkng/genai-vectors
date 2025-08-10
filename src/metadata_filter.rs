use anyhow::Result;
use serde_json::Value;
use std::collections::HashMap;

/// Advanced metadata filtering engine supporting range queries, regex, and boolean logic
#[derive(Debug, Clone)]
pub struct MetadataFilter {
    conditions: Vec<FilterCondition>,
    operator: BooleanOperator,
}

#[derive(Debug, Clone)]
pub enum BooleanOperator {
    And,
    Or,
}

#[derive(Debug, Clone)]
pub enum FilterCondition {
    Equals { field: String, value: Value },
    NotEquals { field: String, value: Value },
    In { field: String, values: Vec<Value> },
    NotIn { field: String, values: Vec<Value> },
    Range { field: String, min: Option<f64>, max: Option<f64> },
    Contains { field: String, substring: String },
    Regex { field: String, pattern: String },
    Exists { field: String },
    NotExists { field: String },
}

impl MetadataFilter {
    pub fn new() -> Self {
        Self {
            conditions: Vec::new(),
            operator: BooleanOperator::And,
        }
    }

    pub fn and(mut self) -> Self {
        self.operator = BooleanOperator::And;
        self
    }

    pub fn or(mut self) -> Self {
        self.operator = BooleanOperator::Or;
        self
    }

    pub fn equals(mut self, field: String, value: Value) -> Self {
        self.conditions.push(FilterCondition::Equals { field, value });
        self
    }

    pub fn not_equals(mut self, field: String, value: Value) -> Self {
        self.conditions.push(FilterCondition::NotEquals { field, value });
        self
    }

    pub fn in_values(mut self, field: String, values: Vec<Value>) -> Self {
        self.conditions.push(FilterCondition::In { field, values });
        self
    }

    pub fn not_in_values(mut self, field: String, values: Vec<Value>) -> Self {
        self.conditions.push(FilterCondition::NotIn { field, values });
        self
    }

    pub fn range(mut self, field: String, min: Option<f64>, max: Option<f64>) -> Self {
        self.conditions.push(FilterCondition::Range { field, min, max });
        self
    }

    pub fn contains(mut self, field: String, substring: String) -> Self {
        self.conditions.push(FilterCondition::Contains { field, substring });
        self
    }

    pub fn regex(mut self, field: String, pattern: String) -> Self {
        self.conditions.push(FilterCondition::Regex { field, pattern });
        self
    }

    pub fn exists(mut self, field: String) -> Self {
        self.conditions.push(FilterCondition::Exists { field });
        self
    }

    pub fn not_exists(mut self, field: String) -> Self {
        self.conditions.push(FilterCondition::NotExists { field });
        self
    }

    /// Apply filter to metadata and return whether it matches
    pub fn matches(&self, metadata: &Value) -> bool {
        if self.conditions.is_empty() {
            return true; // No filter means all match
        }

        let results: Vec<bool> = self.conditions.iter()
            .map(|condition| self.evaluate_condition(condition, metadata))
            .collect();

        match self.operator {
            BooleanOperator::And => results.iter().all(|&x| x),
            BooleanOperator::Or => results.iter().any(|&x| x),
        }
    }

    fn evaluate_condition(&self, condition: &FilterCondition, metadata: &Value) -> bool {
        match condition {
            FilterCondition::Equals { field, value } => {
                self.get_field_value(metadata, field)
                    .map(|v| v == value)
                    .unwrap_or(false)
            }
            FilterCondition::NotEquals { field, value } => {
                self.get_field_value(metadata, field)
                    .map(|v| v != value)
                    .unwrap_or(true)
            }
            FilterCondition::In { field, values } => {
                self.get_field_value(metadata, field)
                    .map(|v| values.contains(v))
                    .unwrap_or(false)
            }
            FilterCondition::NotIn { field, values } => {
                self.get_field_value(metadata, field)
                    .map(|v| !values.contains(v))
                    .unwrap_or(true)
            }
            FilterCondition::Range { field, min, max } => {
                self.get_field_value(metadata, field)
                    .and_then(|v| v.as_f64())
                    .map(|num| {
                        let min_ok = min.map(|m| num >= m).unwrap_or(true);
                        let max_ok = max.map(|m| num <= m).unwrap_or(true);
                        min_ok && max_ok
                    })
                    .unwrap_or(false)
            }
            FilterCondition::Contains { field, substring } => {
                self.get_field_value(metadata, field)
                    .and_then(|v| v.as_str())
                    .map(|s| s.contains(substring))
                    .unwrap_or(false)
            }
            FilterCondition::Regex { field, pattern } => {
                self.get_field_value(metadata, field)
                    .and_then(|v| v.as_str())
                    .map(|s| {
                        // Simple regex matching - in production, use proper regex crate
                        match regex::Regex::new(pattern) {
                            Ok(re) => re.is_match(s),
                            Err(_) => false,
                        }
                    })
                    .unwrap_or(false)
            }
            FilterCondition::Exists { field } => {
                self.get_field_value(metadata, field).is_some()
            }
            FilterCondition::NotExists { field } => {
                self.get_field_value(metadata, field).is_none()
            }
        }
    }

    fn get_field_value<'a>(&self, metadata: &'a Value, field: &str) -> Option<&'a Value> {
        // Support nested field access with dot notation
        let parts: Vec<&str> = field.split('.').collect();
        let mut current = metadata;

        for part in parts {
            current = current.get(part)?;
        }

        Some(current)
    }

    /// Pre-filter IDs before vector search to improve performance
    pub fn pre_filter_ids(&self, metadata_map: &HashMap<String, Value>) -> Vec<String> {
        metadata_map.iter()
            .filter_map(|(id, metadata)| {
                if self.matches(metadata) {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect()
    }
}

impl Default for MetadataFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse filter from JSON query format
impl TryFrom<Value> for MetadataFilter {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self> {
        let mut filter = MetadataFilter::new();

        if let Value::Object(map) = value {
            for (field, condition) in map {
                match &condition {
                    Value::String(_) | Value::Number(_) | Value::Bool(_) => {
                        filter = filter.equals(field, condition);
                    }
                    Value::Object(cond_map) => {
                        for (op, val) in cond_map {
                            filter = match op.as_str() {
                                "$eq" => filter.equals(field.clone(), val.clone()),
                                "$ne" => filter.not_equals(field.clone(), val.clone()),
                                "$in" => {
                                    if let Value::Array(arr) = val {
                                        filter.in_values(field.clone(), arr.clone())
                                    } else {
                                        return Err(anyhow::anyhow!("$in requires array value"));
                                    }
                                }
                                "$nin" => {
                                    if let Value::Array(arr) = val {
                                        filter.not_in_values(field.clone(), arr.clone())
                                    } else {
                                        return Err(anyhow::anyhow!("$nin requires array value"));
                                    }
                                }
                                "$gt" => {
                                    if let Some(num) = val.as_f64() {
                                        filter.range(field.clone(), Some(num + f64::EPSILON), None)
                                    } else {
                                        return Err(anyhow::anyhow!("$gt requires numeric value"));
                                    }
                                }
                                "$gte" => {
                                    if let Some(num) = val.as_f64() {
                                        filter.range(field.clone(), Some(num), None)
                                    } else {
                                        return Err(anyhow::anyhow!("$gte requires numeric value"));
                                    }
                                }
                                "$lt" => {
                                    if let Some(num) = val.as_f64() {
                                        filter.range(field.clone(), None, Some(num - f64::EPSILON))
                                    } else {
                                        return Err(anyhow::anyhow!("$lt requires numeric value"));
                                    }
                                }
                                "$lte" => {
                                    if let Some(num) = val.as_f64() {
                                        filter.range(field.clone(), None, Some(num))
                                    } else {
                                        return Err(anyhow::anyhow!("$lte requires numeric value"));
                                    }
                                }
                                "$contains" => {
                                    if let Some(s) = val.as_str() {
                                        filter.contains(field.clone(), s.to_string())
                                    } else {
                                        return Err(anyhow::anyhow!("$contains requires string value"));
                                    }
                                }
                                "$regex" => {
                                    if let Some(s) = val.as_str() {
                                        filter.regex(field.clone(), s.to_string())
                                    } else {
                                        return Err(anyhow::anyhow!("$regex requires string value"));
                                    }
                                }
                                "$exists" => {
                                    if val.as_bool().unwrap_or(false) {
                                        filter.exists(field.clone())
                                    } else {
                                        filter.not_exists(field.clone())
                                    }
                                }
                                _ => return Err(anyhow::anyhow!("Unknown filter operator: {}", op)),
                            };
                        }
                    }
                    _ => {
                        return Err(anyhow::anyhow!("Invalid filter condition for field: {}", field));
                    }
                }
            }
        }

        Ok(filter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_simple_equals_filter() {
        let filter = MetadataFilter::new()
            .equals("category".to_string(), json!("A"));

        let metadata = json!({"category": "A", "value": 42});
        assert!(filter.matches(&metadata));

        let metadata2 = json!({"category": "B", "value": 42});
        assert!(!filter.matches(&metadata2));
    }

    #[test]
    fn test_range_filter() {
        let filter = MetadataFilter::new()
            .range("score".to_string(), Some(0.5), Some(1.0));

        let metadata1 = json!({"score": 0.7});
        assert!(filter.matches(&metadata1));

        let metadata2 = json!({"score": 0.3});
        assert!(!filter.matches(&metadata2));
    }

    #[test]
    fn test_complex_filter() {
        let filter = MetadataFilter::new()
            .and()
            .equals("category".to_string(), json!("tech"))
            .range("score".to_string(), Some(0.8), None);

        let metadata1 = json!({"category": "tech", "score": 0.9});
        assert!(filter.matches(&metadata1));

        let metadata2 = json!({"category": "tech", "score": 0.7});
        assert!(!filter.matches(&metadata2));
    }

    #[test]
    fn test_nested_field_access() {
        let filter = MetadataFilter::new()
            .equals("user.id".to_string(), json!(123));

        let metadata = json!({
            "user": {
                "id": 123,
                "name": "John"
            }
        });
        assert!(filter.matches(&metadata));
    }
}
