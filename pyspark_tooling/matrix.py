import pyspark.sql.functions as F
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SQLContext
from pyspark.ml.feature import MinHashLSH
from pyspark.mllib.linalg import DenseMatrix
from pyspark.mllib.linalg.distributed import (
    CoordinateMatrix,
    IndexedRow,
    IndexedRowMatrix,
)

from grada_logger import log


def df_to_indexed_row_matrix(row_number_col: str, vector_col: str, df: DataFrame):
    """Convert a dataframe containing a row number and vector to a block matrix"""
    indexed_rows = (
        df.where(F.col(vector_col).isNotNull())
        .select(F.col(row_number_col), F.col(vector_col))
        .rdd.map(
            lambda row: IndexedRow(
                row.__getitem__(row_number_col), row.__getitem__(vector_col).toArray()
            )
        )
    )

    if indexed_rows.isEmpty():
        raise ValueError("Primary RDD is empty. Cannot perform matrix multiplication")

    return IndexedRowMatrix(indexed_rows)


def df_to_block_matrix(row_number_col: str, vector_col: str, df: DataFrame):
    """Convert a dataframe to block matrix"""
    return df_to_indexed_row_matrix(row_number_col, vector_col, df).toBlockMatrix()


def df_to_dense_matrix(row_number_col: str, vector_col: str, df: DataFrame):
    """Convert a dataframe to a local dense matrix"""
    return df_to_block_matrix(row_number_col, vector_col, df).toLocalMatrix()


def multiply_coordinate_matrices(left: CoordinateMatrix, right: CoordinateMatrix):
    """Multiply 2 spark Coordindate Matrices
    without converting either of them into a DenseMatrix.

    NOTE: spark does not provide distributed matrix multiplication of sparse matrices
    for this reason a custom approach has to be used which is discussed here
    https://medium.com/balabit-unsupervised/scalable-sparse-matrix-multiplication-in-apache-spark-c79e9ffc0703
    """

    def key_by_col(x):
        """Take a MatrixEntry of (row, col, val) and
        return a 2-tuple of (col, (row, val))"""
        return (x.j, (x.i, x.value))

    def key_by_row(x):
        """Take a MatrixEntry of (row, col, val) and
        return a 2-tuple of (row, (col, val))"""
        return (x.i, (x.j, x.value))

    left_by_col = left.entries.map(lambda x: key_by_col(x))
    right_by_row = right.entries.map(lambda x: key_by_row(x))

    # Next we perform a row by col matrix multiplication
    # where a shared "key" is used to group entries of the left matrix
    # with COLUMN j and entries of the right matrix with ROW j.
    # Note that entries with the same j will stick together.
    # This should be obvious if you recall that matrix multiplication
    # matches the index of the left column with the index of the right row.
    col_by_row = left_by_col.join(right_by_row)

    def row_by_col_multiplication(x):
        """The input is a key-pair tuple in the following format:
        (key, ((left_row, left_val), (right_col, right_val)))

        the output is a pair of tuples in the following format:
        ((left_row, right_col), (left_val, right_val))

        Note that having finished the grouping we no longer need the shared key anymore,
        (i.e. we no longer need the original indices of the left_col or right_row).
        This is because summed values will go into the output matrix at the
        location (left_row, right_col) and thus we can  regroup by these indices and sum
        """
        return ((x[1][0][0], x[1][1][0]), (x[1][0][1] * x[1][1][1]))

    # multiply elements by the left matrix column and the right matrix row
    products = col_by_row.map(lambda x: row_by_col_multiplication(x))

    # Sum up all the products for the a given left_row and right_col
    summed = products.reduceByKey(lambda accum, n: accum + n)

    # unnest the keys so we can convert back to a coordinate matrix
    flattened = summed.map(lambda x: (x[0][0], x[0][1], x[1]))

    res = CoordinateMatrix(flattened)

    log.info(
        "finished creating coord matrix from dot product",
        rows=res.numRows(),
        cols=res.numCols(),
    )
    return res


def sparse_dot_product_cross_join(
    spark: SQLContext,
    output_col: str,
    primary_row_number_col: str,
    primary_vector_col: str,
    primary_df: DataFrame,
    secondary_row_number_col: str,
    secondary_vector_col: str,
    secondary_df: DataFrame,
):
    """Calculate the dot product for every pair of items between
    a column of SparseVectors in the primary dataframe and a
    column of SparseVectors in the secondary dataframe.

    The input dataframes must have a row number attached. This will
    correspond to the row number in ther resulting row matrix.
    It does not matter if the row numbers are sequential as long
    as they are unique within their dataframes respectively.

    NOTE: if you are using this function in order to generate cosine similarity
    scores then remember to normalize your input vectors first. This way the
    resulting coordinate matrix will represent the similarity scores."""

    def primary_row_to_coords(row):
        """Convert a sparse vector to a list of coords
        in the format of (row_num, col_num, value)"""
        row_num = row.__getitem__(primary_row_number_col)
        vec = row.__getitem__(primary_vector_col)
        return [(row_num, i, j) for i, j in zip(vec.indices, vec.values)]

    primary_rdd = primary_df.select(
        F.col(primary_row_number_col), F.col(primary_vector_col)
    ).rdd.flatMap(lambda row: primary_row_to_coords(row))

    if primary_rdd.isEmpty():
        raise ValueError("Primary RDD is empty. Cannot perform matrix multiplication")

    primary_rdd.persist(StorageLevel.MEMORY_AND_DISK_SER)

    def secondary_row_to_coords(row):
        """Convert a sparse vector to a list of coords
        in the format of (row_num, col_num, value)"""
        row_num = row.__getitem__(secondary_row_number_col)
        vec = row.__getitem__(secondary_vector_col)
        # IMPORTANT - note that we are actually creating
        # the transpose of the secondary matrix hence
        # why the coordinates are back to front
        return [(i, row_num, j) for i, j in zip(vec.indices, vec.values)]

    secondary_rdd = secondary_df.select(
        F.col(secondary_row_number_col), F.col(secondary_vector_col)
    ).rdd.flatMap(lambda row: secondary_row_to_coords(row))

    secondary_rdd.persist(StorageLevel.MEMORY_AND_DISK_SER)

    if secondary_rdd.isEmpty():
        raise ValueError("Secondary RDD is empty. Cannot perform matrix multiplication")

    # create the primary coordinate matrix from the coords
    primary_matrix = CoordinateMatrix(primary_rdd)

    log.info(
        "finished creating primary coordinate matrix",
        rows=primary_matrix.numRows(),
        cols=primary_matrix.numCols(),
    )

    # create the secondary coordinate matrix from the coords
    secondary_matrix = CoordinateMatrix(secondary_rdd)

    log.info(
        "finished creating secondary coordinate matrix transpose",
        rows=secondary_matrix.numRows(),
        cols=secondary_matrix.numCols(),
    )
    coords_matrix = multiply_coordinate_matrices(primary_matrix, secondary_matrix)

    res = coord_matrix_to_dataframe(
        spark,
        primary_row_number_col,
        secondary_row_number_col,
        output_col,
        coords_matrix,
    )

    primary_rdd.unpersist()
    secondary_rdd.unpersist()

    return res


def dense_dot_product_cross_join(
    spark: SQLContext,
    output_col: str,
    primary_row_number_col: str,
    primary_vector_col: str,
    primary_df: DataFrame,
    secondary_row_number_col: str,
    secondary_vector_col: str,
    secondary_df: DataFrame,
):
    """Take """
    primary_matrix = df_to_block_matrix(
        primary_row_number_col, primary_vector_col, primary_df
    )

    primary_matrix.persist(StorageLevel.MEMORY_AND_DISK_SER)

    secondary_matrix = df_to_block_matrix(
        secondary_row_number_col, secondary_vector_col, secondary_df
    ).transpose()

    secondary_matrix.persist(StorageLevel.MEMORY_AND_DISK_SER)

    return dense_matrix_cross_join(
        spark,
        output_col,
        primary_row_number_col,
        primary_matrix,
        secondary_row_number_col,
        secondary_matrix,
    )


def dense_matrix_cross_join(
    spark: SQLContext,
    output_col: str,
    primary_row_number_col: str,
    primary_matrix: IndexedRowMatrix,
    secondary_row_number_col: str,
    secondary_matrix: DenseMatrix,
):
    """Multiply 2 dense matrices to produce a dataframe with pairwise results
    showing primary row number, secondary column number and the dot product as a score
    Note that if you are using this method to produce the cosine similarity of 2 dense
    matrices then it is expected that you have already taken the transpose of the
    secondary matrix"""
    product = primary_matrix.multiply(secondary_matrix)

    log.info(
        "finished dense matrix multiplication",
        num_cols=product.numCols(),
        num_rows=product.numRows(),
    )

    coords_matrix = product.toCoordinateMatrix()

    log.info(
        "finished converting row matrix to coordinate matrix",
        num_cols=coords_matrix.numCols(),
        num_rows=coords_matrix.numRows(),
    )

    return coord_matrix_to_dataframe(
        spark,
        primary_row_number_col,
        secondary_row_number_col,
        output_col,
        coords_matrix,
    )


def coord_matrix_to_dataframe(
    spark: SQLContext,
    primary_row_number_col: str,
    secondary_row_number_col: str,
    output_col: str,
    matrix: CoordinateMatrix,
):

    output_cols = [primary_row_number_col, secondary_row_number_col, output_col]

    return spark.createDataFrame(matrix.entries, output_cols)


def jaccard_cross_join(
    input_col: str,
    output_col: str,
    df: DataFrame,
    primary_df: DataFrame,
    secondary_df: DataFrame,
):
    """Fit a jaccard index model based on all the docs in the corpus.
    Then take a subset of these (the primary docs) and cross join with a different
    subset (the secondary docs) to find any docs that are similar according to the
    minimum similarity specified."""

    hash_col = "hashes"
    min_hash_lsh = MinHashLSH(
        inputCol=input_col, outputCol=hash_col, seed=12345, numHashTables=3
    )
    model = min_hash_lsh.fit(primary_df)

    return model.approxSimilarityJoin(
        primary_df, secondary_df, distCol=output_col, threshold=1.0
    )
