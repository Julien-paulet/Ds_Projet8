#!/usr/bin/env python
# coding: utf-8


import findspark
findspark.init()


import os
import time
from pyspark.sql import SparkSession
from pyspark import SQLContext
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.image import ImageSchema

from pyspark.sql.functions import udf
from pyspark.sql.functions import input_file_name
from pyspark.sql.types import *
from pyspark import SparkContext, SparkConf

from pyspark.ml.image import ImageSchema


# # Settings

os.environ['PYSPARK_DRIVER_PYTHON'] = '/home/ubuntu/anaconda3/envs/OP8/bin/python'
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages com.amazonaws:aws-java-sdk-pom:1.10.34,org.apache.hadoop:hadoop-aws:2.7.2,databricks:spark-deep-learning:1.5.0-spark2.4-s_2.11 pyspark-shell'
os.environ['PYSPARK_PYTHON']='/home/ubuntu/anaconda3/envs/OP8/bin/python'


S3 = True
root_path = ''
path = 'Training/'
bucket = 'op-projet8-julienpaulet3'
categ = ['AppleBraeburn', 'AppleCrimsonSnow']


if S3 is True:
    path_to_save = "s3a://"+bucket+"/"
else:
    path_to_save = root_path


# # Functions

def load_aws_key():
    '''Loading AWS ID and Key from .txt file'''
    with open('Key.txt','r') as f:
        msg = f.read()
    ID = str(msg).split('\n')[0]
    KEY = msg.split('\n')[1]
    os.environ["AWS_ACCESS_KEY_ID"]=ID
    os.environ["AWS_SECRET_ACCESS_KEY"]=KEY
    return ID, KEY

def init_spark_session(S3=False, bucket = '', path_local=''):
    '''SPARK Session setup'''
    spark = SparkSession.builder.master('local[*]').config("spark.driver.memory", "15g").appName('Projet8').getOrCreate()
    sc = spark.sparkContext
    
    if S3 is True:
        path_img = "s3a://"+bucket+"/Training/"
        #Amazon ID and Key
        spark._jsc.hadoopConfiguration().set("fs.s3a.path.style.access", "true")
        spark._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "s3-eu-west-1.amazonaws.com")
        spark._jsc.hadoopConfiguration().set("fs.s3a.aws.access.key", os.environ["AWS_ACCESS_KEY_ID"])
        spark._jsc.hadoopConfiguration().set("fs.s3a.aws.secret.key", os.environ["AWS_SECRET_ACCESS_KEY"])
        # categ = os.listdir(path_img)
    else:
        path_img = path_local
        categ = os.listdir(path)

    return sc, spark, path_img

if S3 is True:
    load_aws_key()

sc, spark, path = init_spark_session(S3=S3,
                                     bucket=bucket,
                                     path_local=path)


def rename_categ(path, categ):
    """Function that transform the path, removing spaces and point"""
    try:
        for cat in categ:
            if ' ' in cat:
                os.rename(os.path.join(path, cat), os.path.join(path, cat.replace(' ', '')))
            elif '.' in cat:
                os.rename(os.path.join(path, cat), os.path.join(path, cat.replace('.', '')))
    except:
        pass
    cat = os.listdir(path)
    return cat

def parse_category(path):
    '''Output image'\ category from the path'''
    if len(path) > 0:
        return path.split('/')[-2]
    else:
        return ''
    
def load_data(path_img):
    '''Df loading ; It takes the path from were the images are and output df with path, images, and categories'''
    
    df_img = spark.read.format("image").load([path + cat for cat in categ])
    #df_img = spark.read.format("image").load(path + categ) # If only one categ
    print('chargement effectué')
    #path from images
    df_img = df_img.withColumn("path", input_file_name())
    #categories
    udf_categorie = udf(parse_category, StringType())
    df_img = df_img.withColumn('categorie', udf_categorie('path'))
    
    return df_img

# # Loading

#Loading of Df with path, images, and categories
spark_df = load_data(path)


# # Preprocessing

# In[11]:


from sparkdl import DeepImageFeaturizer

# We'll use ResNet50 for the transformation
featurizer = DeepImageFeaturizer(inputCol="image", outputCol="image_preprocessed", modelName="ResNet50")
spark_df_preprocessed = featurizer.transform(spark_df).select(['path', 'categorie', 'image_preprocessed'])


# # Saving

#Saving as parquet file
spark_df_preprocessed.repartition(16).write.format("parquet").mode('overwrite').save(path_to_save + 'preprocessed_parquet')

