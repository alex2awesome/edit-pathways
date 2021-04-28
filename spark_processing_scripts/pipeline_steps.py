from pyspark.ml.feature import Normalizer, SQLTransformer
from pyspark.ml.feature import BucketedRandomProjectionLSH
import sparknlp.base as sb
import sparknlp.annotator as sa
from spark_processing_scripts.util_spark import get_word_matching_sql


#####
#
# Pipelines
#
def get_sparknlp_pipeline():
    ####
    #
    # Spark NLP
    #

    documenter = (
        sb.DocumentAssembler()
            .setInputCol("summary")
            .setOutputCol("document")
    )

    sentencer = (
        sa.SentenceDetector()
            .setInputCols(["document"])
            .setOutputCol("sentences")
    )

    tokenizer = (
        sa.Tokenizer()
            .setInputCols(["sentences"])
            .setOutputCol("token")
    )

    word_embeddings = (
        sa.BertEmbeddings
            .load('s3://aspangher/spark_processing_scripts-nlp/small_bert_L4_128_en_2.6.0_2.4')
            .setInputCols(["sentences", "token"])
            .setOutputCol("embeddings")
            .setMaxSentenceLength(512)
            .setBatchSize(100)
    )

    tok_finisher = (
        sb.Finisher()
            .setInputCols(["token"])
            .setIncludeMetadata(True)
    )

    embeddings_finisher = (
        sb.EmbeddingsFinisher()
            .setInputCols("embeddings")
            .setOutputCols("embeddings_vectors")
            .setOutputAsVector(True)
    )

    sparknlp_processing_pipeline = sb.RecursivePipeline(stages=[
        documenter,
        sentencer,
        tokenizer,
        word_embeddings,
        embeddings_finisher,
        tok_finisher
      ]
    )
    return sparknlp_processing_pipeline

def get_explode_pipeline():
    ###
    #
    # SQL Processing Steps
    #
    zip_tok = (
        SQLTransformer()
            .setStatement("""
             SELECT CAST(entry_id AS int) as entry_id,
                    CAST(version AS int) as version, 
                    ARRAYS_ZIP(finished_token, finished_token_metadata, embeddings_vectors) AS zipped_tokens
             FROM __THIS__
        """)
    )

    explode_tok = (
        SQLTransformer()
            .setStatement("""
             SELECT entry_id, version, POSEXPLODE(zipped_tokens) AS (word_idx, zipped_token)
             FROM __THIS__
        """)
    )

    rename_tok = (
        SQLTransformer()
            .setStatement("""
             SELECT entry_id, 
                     version,
                     CAST(zipped_token.finished_token_metadata._2 AS int) AS sent_idx,
                     COUNT(1) OVER(PARTITION BY entry_id, version, zipped_token.finished_token_metadata._2) as num_words,
                     CAST(word_idx AS int) word_idx,
                     zipped_token.finished_token AS token,
                     zipped_token.embeddings_vectors as word_embedding
             FROM __THIS__
        """)
    )
    explode_pipeline = sb.PipelineModel(stages=[
        zip_tok,
        explode_tok,
        rename_tok,
    ])

    return explode_pipeline

def get_similarity_pipeline():
    vector_normalizer = (
        Normalizer(
            inputCol="word_embedding",
            outputCol="norm_word_embedding",
            p=2.0
        )
    )
    similarty_checker = (
        BucketedRandomProjectionLSH(
            inputCol="norm_word_embedding",
            outputCol="hashes",
            bucketLength=3,
            numHashTables=3
        )
    )

    similarity_pipeline = sb.Pipeline(stages=[
        vector_normalizer,
        similarty_checker
    ])

    return similarity_pipeline


def get_sentence_pipelines():
    ## get top sentences, X, pipeline
    s1x, s2x, s3x, s4x = get_word_matching_sql(side='x')
    #
    get_word_pair_min_distance_x = SQLTransformer().setStatement(s1x)
    get_sentence_min_distance_x = SQLTransformer().setStatement(s2x)
    get_min_sentence_x = SQLTransformer().setStatement(s3x)
    threshold_x = SQLTransformer().setStatement(s4x)

    ## get top sentences, Y, pipeline
    s1y, s2y, s3y, s4y = get_word_matching_sql(side='y')
    #
    get_word_pair_min_distance_y = SQLTransformer().setStatement(s1y)
    get_sentence_min_distance_y = SQLTransformer().setStatement(s2y)
    get_min_sentence_y = SQLTransformer().setStatement(s3y)
    threshold_y = SQLTransformer().setStatement(s4y)


    top_sentence_pipeline_x = sb.PipelineModel(stages=[
        get_word_pair_min_distance_x,
        get_sentence_min_distance_x,
        get_min_sentence_x,
        threshold_x
    ])

    top_sentence_pipeline_y = sb.PipelineModel(stages=[
        get_word_pair_min_distance_y,
        get_sentence_min_distance_y,
        get_min_sentence_y,
        threshold_y
    ])

    return top_sentence_pipeline_x, top_sentence_pipeline_y