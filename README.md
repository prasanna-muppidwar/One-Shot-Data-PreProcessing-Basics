# One Shot Data PreProcessing
<div class="cell markdown" id="ASrbCTZRRwUa">

## ***Data Preprocessing:***

  - Data preprocessing is a process of preparing the raw data and making
    it suitable for a machine learning model. It is the first and
    crucial step while creating a machine learning model.
  - When creating a machine learning project, it is not always a case
    that we come across the clean and formatted data. And while doing
    any operation with data, it is mandatory to clean it and put in a
    formatted way. So for this, we use data preprocessing task.
  - For Example we need to process the vegetables before cooking the
    meal same it is here we have to pre process our data before giving
    it to algorithm or ML Model for better output.

</div>

<div class="cell markdown" id="jbBY9RhGIy6_">

Implementation of Basic Python libraries ***Numpy*** and ***Pandas***.

***1. Numpy:***

  - Numpy is a fundamental package for scientific computing and for
    operations on numeric data in Python.
  - It is a python library that provides a multidiemnsional array
    object,various derieved objects ( such as marked arrays and
    matrices) and an assortemnt of routines for fast operations on
    arrays including mathematical, logical ,shape
    manipulation,sorting,selecting, I/o discrete fourier,basic linear
    algebra, basic statistics,random simulation and much more.
  - Numpy maintains minimal memory arrays in numpy are objects.
  - Numpy can be used for shape manipulation and as well as for array
    generation.

</div>

<div class="cell code" data-execution_count="7" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="jYU1XDkZK-fU" data-outputId="59402bf8-d796-4cc5-e5cf-b51e335662c1">

``` python
    import numpy as np
  #Operations on arrays usimg numpy
    ArrayOne = np.array([[1,2,3], [4,5,6], [7,8,9]])         
    ArrayTwo = np.array([[9,8,7], [6,5,4], [3,2,1]])     
    OutputArray = ArrayOne+ArrayTwo
    print(OutputArray)                                                             
```

<div class="output stream stdout">

    [[10 10 10]
     [10 10 10]
     [10 10 10]]

</div>

</div>

<div class="cell code" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="TGYGRnpMLxq0" data-outputId="8084b92a-49df-42d6-fc8a-e5fd2b7f8b4f">

``` python
#arrays of Ones and Zeros
import numpy as np
a = np.identity(10)
b=np.ones(5)
c=np.zeros(5) #here you can also print C

print(b)

```

<div class="output stream stdout">

    [1. 1. 1. 1. 1.]

</div>

</div>

<div class="cell markdown" id="gDzp5UkgTu9q">

***Pandas:***

  - Pandas is an Open Source Python package that is most widely used for
    Data Science and Data Analysis and machine learning Tasks.
  - It is built on top of another package named Numpy which provides
    support for multidimensional arrays and scientific taks.
  - Pandas is a package providing fast flexible and expressive
    DataStructures designed to makeworking with "Relational" or
    "Labelled" data both easy and intuitive.
  - It aims to be the fundamentalhigh level building block for doing
    practical , real-world data analysis in Puthon.

</div>

<div class="cell code" data-colab="{&quot;height&quot;:206,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="TT7DyfTSTqiy" data-outputId="f8ec8039-5560-4f24-e5a0-25665ea082cd">

``` python
#here with the help of pandas we created a toy data set for future processing
import pandas as pd
df = pd.DataFrame({
    "Serial No":[1,2,3,4,5],
    "First Name":["Prasanna","Aman","Lokesh","Animesh","Naman"],
    "Lats Name":["Muppidwar","Behla","Jain","Agarwal","Mathur"],
    "Age":[19,23,30,26,25]
})

df
```

<div class="output execute_result" data-execution_count="9">

``` 
   Serial No First Name  Lats Name  Age
0          1   Prasanna  Muppidwar   19
1          2       Aman      Behla   23
2          3     Lokesh       Jain   30
3          4    Animesh    Agarwal   26
4          5      Naman     Mathur   25
```

</div>

</div>

<div class="cell markdown" id="T2iSdDVnWNBZ">

***Missing Data:***

  - Missing Data is the value or entity that is not present or recorded
    into a DataSet. These values can be Single Missing Value in a cell
    or it can be a entire Missing Observation (row).
  - Missing data can occur both in Continous Variable( eg. Age of
    Student) or Categorial variable such as Gender of Population.
  - In progrsmming language such as Python missing values are
    represented as "Na" or "nan" or simply as empty cell i a row or
    column.
  - For Example: You are Administrating a Survey in college and a
    student have given a false statement or is unable to provide
    information, now what you have to keep the feild empty as you cannot
    Enter False Data.
  - Missing Values are definately undesirable but its difficult to
    qualify the magnitable of efforts in statisticals and ML projects.
  - If its large data set and a very small percentage of data is missing
    the effect may not be detachable at all. However if the data set is
    relatively small every data point counts.

</div>

<div class="cell code" data-colab="{&quot;height&quot;:206,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="-f1sNPZYbqwV" data-outputId="9acf8f3a-84d6-4b99-d61d-8cb0a3f07247">

``` python
#Lets create a toy data set with null values and try filling it.
import pandas as pd 
import pandas as pd
df = pd.DataFrame({
    "Serial No":[1,2,3,4,5],
    "First Name":["Prasanna","Aman","Lokesh","Animesh","Naman"],
    "Lats Name":["Muppidwar","Behla","Jain","Agarwal","Mathur"],
    "Age":[19,23,30,np.nan,25]})
#lets assume we have a empty space in the data set and we have to fill it 
df
```

<div class="output execute_result" data-execution_count="10">

``` 
   Serial No First Name  Lats Name   Age
0          1   Prasanna  Muppidwar  19.0
1          2       Aman      Behla  23.0
2          3     Lokesh       Jain  30.0
3          4    Animesh    Agarwal   NaN
4          5      Naman     Mathur  25.0
```

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:206,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="irfyarn5ckJK" data-outputId="7d101a3c-85c2-4ca0-f5cd-fd5e3ca8d75a">

``` python
#now we have to fill the data using mathematical calculations we cant add imaginary data here
df['Age'].fillna(round(df['Age'].mean()),inplace=True)
df
```

<div class="output execute_result" data-execution_count="11">

``` 
   Serial No First Name  Lats Name   Age
0          1   Prasanna  Muppidwar  19.0
1          2       Aman      Behla  23.0
2          3     Lokesh       Jain  30.0
3          4    Animesh    Agarwal  24.0
4          5      Naman     Mathur  25.0
```

</div>

</div>

<div class="cell markdown" id="aAuE5VbkdaOU">

Here We have successfully filled the null or empty space using following
functions

  - fillna() : which is used to fill null values.
  - mean() : to find mean of given data \*round() : to convert a
    floating int to int

</div>

<div class="cell code" data-colab="{&quot;height&quot;:645,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="3KaAOHYGd131" data-outputId="845462cd-191c-4142-fbaf-756377c9c069">

``` python
#here we will be using toy data of indian cricket team for processing
import pandas as pd
import numpy as np

df=pd.DataFrame({
    #using indian team data
    "Name":['R Dravid','R Sharma','S Dhawan','V Kohli','S Yadav','S Iyyer','R Ashwin',
            'R Jadeja','M Shami','B Kumar','Y Chahal','R Pant','W Saha','C Pujara','M Pandey',
            'S Samson','K Nair','M Agarwal','K Yadav'],
    "Age":[50,np.nan,35,30,37,25,26,17,18,29,23,24,24,24,26,26,30,26,29],
    "Role":["Head Coach","Bat","Bat","Bat","Bat","Bat","Bat","All","All","Ball",
            "Ball","Ball","Bat","Bat","Bat","Bat","Bat","Bat","Ball"],
    "Hand":["Right","Right","Left","Right","Right","Right",np.nan,"Right","Right",
            "Right","Right","Right","Right","Right","Right","Right","Right","Right","Right"]})
df
```

<div class="output execute_result" data-execution_count="17">

``` 
         Name   Age        Role   Hand
0    R Dravid  50.0  Head Coach  Right
1    R Sharma   NaN         Bat  Right
2    S Dhawan  35.0         Bat   Left
3     V Kohli  30.0         Bat  Right
4     S Yadav  37.0         Bat  Right
5     S Iyyer  25.0         Bat  Right
6    R Ashwin  26.0         Bat    NaN
7    R Jadeja  17.0         All  Right
8     M Shami  18.0         All  Right
9     B Kumar  29.0        Ball  Right
10   Y Chahal  23.0        Ball  Right
11     R Pant  24.0        Ball  Right
12     W Saha  24.0         Bat  Right
13   C Pujara  24.0         Bat  Right
14   M Pandey  26.0         Bat  Right
15   S Samson  26.0         Bat  Right
16     K Nair  30.0         Bat  Right
17  M Agarwal  26.0         Bat  Right
18    K Yadav  29.0        Ball  Right
```

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:645,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="0GpCGMMeebsy" data-outputId="3adaa8af-72ef-4f2a-e787-891199b4e501">

``` python
#To Find Age
df['Age'].fillna(round(df["Age"].mean()),inplace=True) #round for type casting
#To Inplace Value manually
df['Hand'].fillna(df["Hand"].mode(),inplace=True)
#To Inplace value manually
df['Role'].fillna(df["Role"].mode(),inplace=True)

df
```

<div class="output execute_result" data-execution_count="21">

``` 
         Name   Age        Role   Hand
0    R Dravid  50.0  Head Coach  Right
1    R Sharma  28.0         Bat  Right
2    S Dhawan  35.0         Bat   Left
3     V Kohli  30.0         Bat  Right
4     S Yadav  37.0         Bat  Right
5     S Iyyer  25.0         Bat  Right
6    R Ashwin  26.0         Bat    NaN
7    R Jadeja  17.0         All  Right
8     M Shami  18.0         All  Right
9     B Kumar  29.0        Ball  Right
10   Y Chahal  23.0        Ball  Right
11     R Pant  24.0        Ball  Right
12     W Saha  24.0         Bat  Right
13   C Pujara  24.0         Bat  Right
14   M Pandey  26.0         Bat  Right
15   S Samson  26.0         Bat  Right
16     K Nair  30.0         Bat  Right
17  M Agarwal  26.0         Bat  Right
18    K Yadav  29.0        Ball  Right
```

</div>

</div>

<div class="cell markdown" id="jra2ft67Lni-">

***Normalization:***

  - Normalization is a technique used to scale the values of a dataset
    to a specific range, such as 0 to 1.
  - This is often done to ensure that all features in a dataset are on a
    similar scale, as some machine learning algorithms can be sensitive
    to the scale of the input features.
  - In Python, the most common library used for normalization is
    scikit-learn, which provides a number of preprocessing functions,
    including the MinMaxScaler class, which can be used to normalize a
    dataset.
  - Another library is pandas which provide function like
    minmax\_scale() to normalize the data. It is important to note that
    normalization should only be applied to the training set, and then
    the same scaling parameters should be used to transform the test set
    or any new data.

</div>

<div class="cell code" data-execution_count="1" data-colab="{&quot;height&quot;:206,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="_fsT296MeerR" data-outputId="6d1e7996-9796-4180-feaf-e08827e03735">

``` python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.DataFrame({
    "Serial No":[1,2,3,4,5],
    "First Name":["Prasanna","Aman","Lokesh","Animesh","Naman"],
    "Lats Name":["Muppidwar","Behla","Jain","Agarwal","Mathur"],
    "Age":[19,23,30,24,25]})
df
```

<div class="output execute_result" data-execution_count="1">

``` 
   Serial No First Name  Lats Name  Age
0          1   Prasanna  Muppidwar   19
1          2       Aman      Behla   23
2          3     Lokesh       Jain   30
3          4    Animesh    Agarwal   24
4          5      Naman     Mathur   25
```

</div>

</div>

<div class="cell code" data-execution_count="3" data-colab="{&quot;height&quot;:206,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="lWBzjLoYNAql" data-outputId="8fb2fc77-0c7c-45be-837c-1f3f35dbac10">

``` python
scale = MinMaxScaler()
df[['Age']] = scale.fit_transform(df[['Age']]) 

df
```

<div class="output execute_result" data-execution_count="3">

``` 
   Serial No First Name  Lats Name       Age
0          1   Prasanna  Muppidwar  0.000000
1          2       Aman      Behla  0.363636
2          3     Lokesh       Jain  1.000000
3          4    Animesh    Agarwal  0.454545
4          5      Naman     Mathur  0.545455
```

</div>

</div>

<div class="cell markdown" id="6MwsVTqgNd1P">

***Standardization:*** Standardization is a technique used to transform
a dataset so that it has a mean of 0 and a standard deviation of 1. This
is often done to ensure that all features in a dataset have similar
properties, as some machine learning algorithms assume that the input
features are normally distributed. In Python, the most common library
used for standardization is scikit-learn, which provides a number of
preprocessing functions, including the StandardScaler class, which can
be used to standardize a dataset. Another library is pandas which
provide function like scale() to standardize the data. It is important
to note that standardization should only be applied to the training set,
and then the same scaling parameters should be used to transform the
test set or any new data. Also, it is important to standardize only
numerical variables if the dataset contains categorical variables it may
not be suitable to standardize.

</div>

<div class="cell code" data-execution_count="4" data-colab="{&quot;height&quot;:206,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="Sic-2GRLOEPH" data-outputId="268187e3-bf43-4a3d-c835-40e3c7fefa0f">

``` python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.DataFrame({
    "Serial No":[1,2,3,4,5],
    "First Name":["Prasanna","Aman","Lokesh","Animesh","Naman"],
    "Lats Name":["Muppidwar","Behla","Jain","Agarwal","Mathur"],
    "Age":[19,23,30,24,25]})
df
```

<div class="output execute_result" data-execution_count="4">

``` 
   Serial No First Name  Lats Name  Age
0          1   Prasanna  Muppidwar   19
1          2       Aman      Behla   23
2          3     Lokesh       Jain   30
3          4    Animesh    Agarwal   24
4          5      Naman     Mathur   25
```

</div>

</div>

<div class="cell code" data-execution_count="6" data-colab="{&quot;height&quot;:206,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="RfRyl2wKOO2l" data-outputId="64ed0ef7-f354-4598-fa44-459214d82d26">

``` python
scale = StandardScaler()
df[['Age']] = scale.fit_transform(df[['Age']]) 
df
```

<div class="output execute_result" data-execution_count="6">

``` 
   Serial No First Name  Lats Name       Age
0          1   Prasanna  Muppidwar -1.467265
1          2       Aman      Behla -0.338600
2          3     Lokesh       Jain  1.636565
3          4    Animesh    Agarwal -0.056433
4          5      Naman     Mathur  0.225733
```

</div>

</div>

<div class="cell code" id="sA6Sw8BgOhOQ">

``` python
```

</div>
