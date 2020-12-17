import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
df = pd.read_csv(
    '/content/drive/My Drive/Colab Notebooks/data/data_with_category.csv')
plt.style.use('ggplot')

plt.figure(figsize=(14, 16))
df['job_type_name'].value_counts().plot(kind='barh')
plt.show()

plt.figure(figsize=(16, 5))
df['job_type_category_name'].value_counts().plot(kind='barh')
plt.show()
