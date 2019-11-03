
# coding: utf-8

# In[131]:


import matplotlib.pyplot as plt


# In[132]:


from fcmeans import FCM


# In[133]:


pic = plt.imread('1.jpeg')/255  


# In[134]:


print(pic1.shape)
plt.imshow(pic)


# In[135]:


fcm = FCM(n_clusters=3)


# In[136]:


pic_n = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2])
pic_n.shape


# In[137]:


fcm.fit(pic_n)


# In[138]:


fcm_centers = fcm.centers
fcm_labels  = fcm.u.argmax(axis=1)


# In[139]:


pic2show = fcm_centers[fcm_labels]


# In[140]:


cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2])
plt.imshow(cluster_pic)

