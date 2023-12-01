# from langchain.embeddings import HuggingFaceHubEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.text_splitter import CharacterTextSplitter
# from langchain import HuggingFaceHub

# import os
# os.environ['HUGGINGFACEHUB_API_TOKEN']='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'

# numpy="""NumPy, short for Numerical Python, is a fundamental library in the Python ecosystem that is widely used for numerical and scientific computing. It provides support for large, multi-dimensional arrays and matrices, along with a collection of high-level mathematical functions to operate on these arrays. NumPy is an essential tool in the toolkit of data scientists, engineers, and researchers who deal with numerical data and perform complex mathematical computations.

# Use Cases of NumPy

# NumPy has a wide range of use cases, making it an integral part of various domains and applications:

# Data Manipulation: NumPy arrays serve as the building blocks for handling structured data. They enable data cleaning, transformation, and efficient analysis.

# Mathematical and Statistical Operations: NumPy offers a plethora of functions for performing mathematical and statistical operations. You can easily calculate mean, median, standard deviation, and more.

# Linear Algebra: The library provides robust support for linear algebra operations. You can perform tasks such as matrix multiplication, eigenvalue calculations, and solving linear systems of equations.

# Signal Processing: In signal processing, NumPy is a crucial component for filtering and Fourier transformations. It plays a vital role in applications like audio processing and image analysis.

# Machine Learning: NumPy forms the backbone for many machine learning libraries like scikit-learn and TensorFlow. It allows efficient storage and manipulation of data, which is crucial for training machine learning models.

# Image Processing: Libraries like OpenCV heavily rely on NumPy for image manipulation and analysis. NumPy's array operations make it an ideal choice for working with pixel data.

# Simulation and Modeling: Scientists and engineers use NumPy for simulating physical phenomena and creating mathematical models. It's indispensable in fields such as physics, chemistry, and engineering.

# Creating NumPy Arrays

# NumPy arrays are at the core of the library's functionality. To work with NumPy, you need to create arrays, which can be done in several ways:

# You can create a one-dimensional array using the np.array() function, passing a Python list as an argument. For example:
# python

# import numpy as np

# arr = np.array([1, 2, 3, 4, 5])
# For two-dimensional arrays (matrices), you can use the np.array() function with a nested list:
# python

# matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# Important NumPy Functions

# NumPy provides a wide array of functions to work with arrays. Some of the fundamental functions include:

# np.zeros(shape): Creates an array filled with zeros of the specified shape. For example:
# python

# zero_array = np.zeros((3, 3))
# np.ones(shape): Generates an array filled with ones. Here's an example:

# ones_array = np.ones((2, 4))
# np.empty(shape): Creates an empty array without initializing its values. For instance:
# python

# empty_array = np.empty((2, 2))
# np.add(arr1, arr2): Adds two arrays element-wise. This function is handy for element-wise operations:
# python

# result = np.add(arr1, arr2)
# np.dot(arr1, arr2): Performs matrix multiplication. It's invaluable for linear algebra operations:
# python

# result = np.dot(arr1, arr2)
# np.mean(arr): Calculates the mean of the array. You can use this to find the average value of an array:
# python

# average = np.mean(arr)
# np.std(arr): Computes the standard deviation of the array. This function is useful for assessing the spread of data:
# python

# std_dev = np.std(arr)
# Array Slicing and Indexing

# NumPy provides powerful tools for slicing and indexing arrays:

# arr[start:stop:step]: This allows you to slice an array. You can specify the starting index, stopping index, and step size:
# python

# sliced = arr[1:4]
# arr[index]: To access specific elements within an array, you can use indexing:
# python

# element = arr[2]
# Shape Manipulation

# Array shapes and dimensions can be easily manipulated using NumPy functions:

# arr.shape: To get the shape of an array (its dimensions), you can access the shape attribute:
# python

# shape = arr.shape
# arr.reshape(new_shape): This function allows you to reshape an array. You specify the desired shape as the argument:
# python

# reshaped = arr.reshape((2, 3))
# np.transpose(arr): Transposing an array swaps its rows and columns, effectively flipping it:
# python

# transposed = np.transpose(arr)
# Broadcasting

# NumPy supports broadcasting, which allows you to perform operations on arrays of different shapes. This simplifies code and reduces the need for explicit loops.

# For example, you can add a scalar to an array, and NumPy will broadcast the scalar to match the array's shape:

# python

# arr = np.array([1, 2, 3])
# result = arr + 2  # Broadcasting the scalar to the array

# Certainly, there's much more to explore about NumPy. In this extended 1000-word text, we will delve into more advanced topics and features of NumPy, as well as some tips and best practices for using the library effectively.

# **Advanced NumPy Features**

# 1. **Fancy Indexing**: NumPy allows you to index arrays using arrays of integers or boolean values. This is called fancy indexing and can be a powerful tool for data manipulation. For example:

# ```python
# arr = np.array([1, 2, 3, 4, 5])
# indices = np.array([0, 2, 4])
# subset = arr[indices]  # Selects elements at indices 0, 2, and 4
# ```

# 2. **Broadcasting**: We've mentioned broadcasting before, but it's worth exploring in more detail. Broadcasting allows NumPy to perform operations on arrays with different shapes. For instance, you can add a 1D array to a 2D array, and NumPy will automatically extend the 1D array to match the shape of the 2D array.

# ```python
# a = np.array([[1, 2, 3], [4, 5, 6]])
# b = np.array([10, 20, 30])
# result = a + b  # Broadcasting b to match the shape of a
# ```

# 3. **Universal Functions (ufuncs)**: NumPy provides a wide range of universal functions that operate element-wise on arrays. These functions are highly optimized and allow for efficient computations. Examples include `np.sin()`, `np.exp()`, and `np.log()`.

# ```python
# arr = np.array([0, np.pi/2, np.pi])
# sine_values = np.sin(arr)  # Calculates the sine of each element
# ```

# 4. **Array Concatenation and Splitting**: You can concatenate arrays using functions like `np.concatenate()`, `np.vstack()`, and `np.hstack()`. Conversely, you can split arrays using functions like `np.split()` and `np.hsplit()`.

# ```python
# array1 = np.array([1, 2, 3])
# array2 = np.array([4, 5, 6])
# concatenated = np.concatenate((array1, array2))  # Concatenates the two arrays
# ```

# 5. **Element-wise Comparison**: NumPy allows you to perform element-wise comparisons between arrays, resulting in Boolean arrays. This is useful for tasks like filtering data.

# ```python
# arr = np.array([1, 2, 3, 4, 5])
# condition = arr > 2
# filtered_arr = arr[condition]  # Selects elements greater than 2
# ```

# 6. **File Input and Output**: NumPy provides functions for efficiently reading and writing array data to files. You can use `np.save()` and `np.load()` to store and retrieve NumPy arrays.

# ```python
# arr = np.array([1, 2, 3, 4, 5])
# np.save('my_array.npy', arr)  # Save the array to a file
# loaded_arr = np.load('my_array.npy')  # Load the array from the file
# ```

# 7. **Random Number Generation**: NumPy has a random module (`np.random`) that allows you to generate random numbers, samples, and distributions. This is valuable for tasks like simulation and statistical analysis.

# ```python
# random_numbers = np.random.rand(5)  # Generate an array of 5 random numbers between 0 and 1
# ```

# **Best Practices**

# 1. **Vectorized Operations**: NumPy is highly optimized for vectorized operations. Whenever possible, avoid explicit loops and utilize NumPy's functions to operate on entire arrays. This leads to faster and more efficient code.

# 2. **Memory Usage**: Be mindful of memory usage, especially when working with large datasets. NumPy arrays can consume a significant amount of memory. Consider using data types with lower memory footprints when appropriate.

# 3. **Array Shape and Dimensionality**: Understanding the shape and dimension of arrays is crucial. Use functions like `shape`, `reshape`, and `transpose` to manipulate arrays to suit your needs.

# 4. **Use ufuncs**: Leveraging universal functions (ufuncs) can significantly improve the performance of your code. NumPy's ufuncs are highly optimized and execute faster than equivalent Python loops.

# 5. **NumPy Documentation**: NumPy has extensive documentation with examples and explanations of its functions. When in doubt, refer to the official documentation to learn more about specific functions and their usage.

# 6. **Pandas Integration**: NumPy plays well with Pandas, another essential library for data analysis. You can easily convert NumPy arrays to Pandas DataFrames and vice versa, allowing you to take advantage of both libraries' strengths.

# 7. **NumPy in Multidisciplinary Fields**: NumPy is not exclusive to any single domain. It's a versatile tool that is used in fields ranging from economics and finance to physics and biology. Familiarity with NumPy can open doors in a wide range of disciplines.

# NumPy is a versatile and indispensable library for numerical and scientific computing in Python. This extended text has covered more advanced features and best practices, expanding on the previous overview. With NumPy, you can efficiently work with large datasets, perform complex mathematical operations, and tackle a variety of scientific and engineering problems. Its array manipulation capabilities, broadcasting, and universal functions make it a go-to tool for professionals and researchers across different fields. To master NumPy, practice is key. Experiment with the library, explore its extensive documentation, and continue learning about its features to become proficient in scientific computing with Python.

# Conclusion

# NumPy is a versatile and powerful library that plays a crucial role in scientific computing and data analysis with Python. Its extensive capabilities in creating and manipulating arrays, performing mathematical operations, and supporting various use cases make it an indispensable tool for researchers, data scientists, engineers, and developers. This 1000-word text provides an overview of NumPy's significance, core functions, and diverse applications, but it only scratches the surface of what this library can offer. Understanding NumPy is fundamental for anyone dealing with numerical data in Python and is an essential step toward becoming proficient in scientific computing and data analysis.
# """






# # Split the text data
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# # Create documents as per LangChain schema
# texts = text_splitter.create_documents([numpy])

# # initialize the embedding strategy
# embeddings = HuggingFaceHubEmbeddings()

# # Convert documents to embeddings
# docsearch = Chroma.from_documents(texts, embeddings)



# repo_id = "google/flan-t5-xxl"

# # Repo from HuggingFaceHub
# flan_t5 = HuggingFaceHub(repo_id=repo_id,
#                          model_kwargs={"temperature":0.1, "max_new_tokens":200})


# Create the LLM Chain
import chat_model
from langchain.chains import RetrievalQA

flan_t5_qa = RetrievalQA.from_chain_type(llm = chat_model.flan_t5, chain_type = "stuff",
                                            retriever = chat_model.docsearch.as_retriever(),
                                            return_source_documents = True)

def generate_response(query):
    # query = "what does numpy provide support for?"
    response = flan_t5_qa(query)
    # print(response["result"])
    return str(response["result"])