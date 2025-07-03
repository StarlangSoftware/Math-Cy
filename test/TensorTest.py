import unittest
from Math.Tensor import Tensor


class TensorTest(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create test tensors
        self.tensor1d = Tensor([1.0, 2.0, 3.0, 4.0])
        self.tensor2d = Tensor([[1.0, 2.0], [3.0, 4.0]])
        self.tensor3d = Tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        
        # Create tensors for operations
        self.tensor_a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        self.tensor_b = Tensor([[5.0, 6.0], [7.0, 8.0]])
        self.tensor_scalar = Tensor([2.0])  # For broadcasting tests

    def test_initialization(self):
        """Test tensor initialization with different shapes."""
        # Test 1D tensor
        self.assertEqual(self.tensor1d.shape, (4,))
        self.assertEqual(self.tensor1d.get((0,)), 1.0)
        self.assertEqual(self.tensor1d.get((3,)), 4.0)
        
        # Test 2D tensor
        self.assertEqual(self.tensor2d.shape, (2, 2))
        self.assertEqual(self.tensor2d.get((0, 0)), 1.0)
        self.assertEqual(self.tensor2d.get((1, 1)), 4.0)
        
        # Test 3D tensor
        self.assertEqual(self.tensor3d.shape, (2, 2, 2))
        self.assertEqual(self.tensor3d.get((0, 0, 0)), 1.0)
        self.assertEqual(self.tensor3d.get((1, 1, 1)), 8.0)

    def test_get_set(self):
        """Test get and set operations."""
        tensor = Tensor([[1.0, 2.0], [3.0, 4.0]])
        
        # Test get
        self.assertEqual(tensor.get((0, 0)), 1.0)
        self.assertEqual(tensor.get((1, 1)), 4.0)
        
        # Test set
        tensor.set((0, 1), 10.0)
        self.assertEqual(tensor.get((0, 1)), 10.0)

    def test_add(self):
        """Test tensor addition."""
        result = self.tensor_a.add(self.tensor_b)
        
        expected = Tensor([[6.0, 8.0], [10.0, 12.0]])
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.get((0, 0)), 6.0)
        self.assertEqual(result.get((1, 1)), 12.0)

    def test_subtract(self):
        """Test tensor subtraction."""
        result = self.tensor_b.subtract(self.tensor_a)
        
        expected = Tensor([[4.0, 4.0], [4.0, 4.0]])
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.get((0, 0)), 4.0)
        self.assertEqual(result.get((1, 1)), 4.0)

    def test_hadamardProduct(self):
        """Test element-wise multiplication (Hadamard product)."""
        result = self.tensor_a.hadamardProduct(self.tensor_b)
        
        expected = Tensor([[5.0, 12.0], [21.0, 32.0]])
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.get((0, 0)), 5.0)
        self.assertEqual(result.get((1, 1)), 32.0)

    def test_multiply(self):
        """Test matrix multiplication."""
        result = self.tensor_a.multiply(self.tensor_b)
        
        # Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.get((0, 0)), 19.0)
        self.assertEqual(result.get((0, 1)), 22.0)
        self.assertEqual(result.get((1, 0)), 43.0)
        self.assertEqual(result.get((1, 1)), 50.0)

    def test_broadcast_to(self):
        """Test broadcasting functionality."""
        # Test broadcasting scalar to 2D
        scalar = Tensor([5.0])
        result = scalar.broadcast_to((2, 2))
        
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.get((0, 0)), 5.0)
        self.assertEqual(result.get((1, 1)), 5.0)
        
        # Test broadcasting 1D to 2D
        vector = Tensor([1.0, 2.0])
        result = vector.broadcast_to((3, 2))
        
        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result.get((0, 0)), 1.0)
        self.assertEqual(result.get((0, 1)), 2.0)
        self.assertEqual(result.get((2, 0)), 1.0)
        self.assertEqual(result.get((2, 1)), 2.0)

    def test_reshape(self):
        """Test tensor reshaping."""
        # Reshape 2x2 to 1x4
        result = self.tensor2d.reshape((1, 4))
        self.assertEqual(result.shape, (1, 4))
        self.assertEqual(result.get((0, 0)), 1.0)
        self.assertEqual(result.get((0, 3)), 4.0)
        
        # Reshape 1x4 back to 2x2
        result = result.reshape((2, 2))
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.get((0, 0)), 1.0)
        self.assertEqual(result.get((1, 1)), 4.0)

    def test_transpose(self):
        """Test tensor transpose."""
        # Test 2D transpose
        result = self.tensor2d.transpose()
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.get((0, 0)), 1.0)
        self.assertEqual(result.get((0, 1)), 3.0)
        self.assertEqual(result.get((1, 0)), 2.0)
        self.assertEqual(result.get((1, 1)), 4.0)
        
        # Test 3D transpose with custom axes
        result = self.tensor3d.transpose((2, 0, 1))
        self.assertEqual(result.shape, (2, 2, 2))
        self.assertEqual(result.get((0, 0, 0)), 1.0)
        self.assertEqual(result.get((1, 1, 1)), 8.0)

    def test_partial(self):
        """Test partial tensor extraction."""
        # Extract a 1x1 sub-tensor from 2x2
        result = self.tensor2d.partial((0, 0), (1, 1))
        self.assertEqual(result.shape, (1, 1))
        self.assertEqual(result.get((0, 0)), 1.0)
        
        # Extract a 1x2 sub-tensor from 2x2
        result = self.tensor2d.partial((0, 0), (1, 2))
        self.assertEqual(result.shape, (1, 2))
        self.assertEqual(result.get((0, 0)), 1.0)
        self.assertEqual(result.get((0, 1)), 2.0)

    def test_broadcasting_operations(self):
        """Test operations with broadcasting."""
        # Add scalar to tensor
        scalar = Tensor([10.0])
        result = self.tensor2d.add(scalar)
        
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.get((0, 0)), 11.0)
        self.assertEqual(result.get((1, 1)), 14.0)
        
        # Multiply tensor by scalar
        result = self.tensor2d.hadamardProduct(scalar)
        self.assertEqual(result.get((0, 0)), 10.0)
        self.assertEqual(result.get((1, 1)), 40.0)

    def test_error_handling(self):
        """Test error handling for invalid operations."""
        # Test invalid shape for reshape
        with self.assertRaises(ValueError):
            self.tensor2d.reshape((3, 3))  # Wrong number of elements
        
        # Test invalid indices
        with self.assertRaises(IndexError):
            self.tensor2d.get((2, 2))  # Out of bounds
        
        # Test invalid matrix multiplication
        tensor_3x2 = Tensor([[1, 2], [3, 4], [5, 6]])
        with self.assertRaises(ValueError):
            self.tensor2d.multiply(tensor_3x2)  # Shape mismatch

    def test_complex_operations(self):
        """Test more complex tensor operations."""
        # Create a 3D tensor and perform operations
        tensor_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        
        # Reshape to 2D
        tensor_2d = tensor_3d.reshape((4, 2))
        self.assertEqual(tensor_2d.shape, (4, 2))
        self.assertEqual(tensor_2d.get((0, 0)), 1.0)
        self.assertEqual(tensor_2d.get((3, 1)), 8.0)
        
        # Transpose and multiply
        tensor_2d_t = tensor_2d.transpose()
        result = tensor_2d.multiply(tensor_2d_t)
        self.assertEqual(result.shape, (4, 4))

    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.tensor2d)
        self.assertIn("Tensor", repr_str)
        self.assertIn("shape=(2, 2)", repr_str)
        self.assertIn("data=", repr_str)


if __name__ == '__main__':
    unittest.main() 