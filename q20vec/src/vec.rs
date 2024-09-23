//! Vector Embeddings and vector operations

/// A `DIM` dimensional Vector with a value at it's tip
pub struct EmbeddingVector<VAL, const DIMS: usize> {
    /// The tip-held value
    pub value: Option<VAL>,
    /// The individual components in each dimension of this vector
    components: [f32; DIMS],
}

impl<VAL, const DIMS: usize> std::ops::Mul for EmbeddingVector<VAL, DIMS> {
    type Output = f32;
    fn mul(self, rhs: Self) -> Self::Output {
        self.dot_product(&rhs)
    }
}

impl<VAL, const DIMS: usize> EmbeddingVector<VAL, DIMS> {
    /// Creates the unit vector in `DIMS` dimensions
    pub fn unit_vector() -> Self {
        Self {
            components: [1f32; DIMS],
            value: None,
        }
    }

    /// Performs the dot product on two vectors (values are ignored)
    pub fn dot_product(&self, other: &EmbeddingVector<VAL, DIMS>) -> f32 {
        self.components
            .iter()
            .zip(other.components.iter())
            .map(|(a1, a2)| a1 + a2)
            .sum()
    }
}
