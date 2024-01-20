# Syn-Aesthetics

https://medium.com/@JesseGray/aesthetic-dysphoria-asking-for-directions-in-the-uncanny-valley-c323e3b79776

Table of Contents:

Overview

Description(long story short)

Key features

Installation

Usage

License

Contributions

Contact

Overview:
A substantial body of research and development has been invested into studying the effects of the 'uncanny valley' in film, animation, and visual media, but much less attention has been paid to how these reactions relate to other media that use AI algorithms to approximate human observation and interaction. Within these emerging technologies is the potential presence of an inherent stress-inducing quality in LLM-driven writing, speech, music, and chat models, possibly even extending to self-driving cars. Could regularly interacting with a generative pre-trained model instill a measurable sense of dysphoria?

This project seeks to explore that question through examining several datasets associated with the ChillsDB database, a collection of responses to chill inducing stimuli including film, speech, and music. By establishing a threshhold for the physical effects of chills, and qualifying them as a good or bad phenomenon, predictive models can be developed to prescreen other media for potential subliminal negative emotional response. 

While not perhaps the most dramatic or dangerous example of generalized artificial intelligence backfiring, the off-putting quality of the uncanny valley is a substantial obstacle in the adoption of these tools. If not analyzed more effectively and countered,  it could lead generative pretrained models to go the way of Zoltar. 

Description:

Data-driven study of the 'uncanny valley' effect in AI-generated media, and its implications on AI adoption

Key Features:

🍎🍊 Comprehensive analysis of various datasets from ChillsDB.


⛏🔦 Detailed exploration of the 'uncanny valley' effect


🖼 🎨 Insightful visualizations of the physical and emotional responses to AI-generated media.


🤖🎢 Innovative approach to engineering more concordant AI models.

Installation:

I wanted to keep this project  lightweight and flexible to maximize access and focus on EDA instead of building a toolkit to address the question. Google Colaboratory is an open-source Jupyter notebook that offers user-friendly framework for data science, and internally supports a diverse range of packages and libraries. It's especially good as as a scratchbook to test models and run multiple approaches to see which one works best for the datset at hand. Colab is kind of a sleeper hit, but it's been invaluable to me as a problem-solving tool. More info can be found here: 

https://colab.research.google.com/

and there is a link at the top of each notebook in the repository🙂

The datasets we are working with are available in this repository as well, or via Harvard dataverse here: 

https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ADLSZE 

along with several other ChillsDb datasets that are not yet part of this project.


Standard Import Library for EDA and data viz:

![Screen Shot 2024-01-20 at 2 31 18 PM](https://github.com/JessemanGray/Syn-Aesthetics/assets/123507565/80ffc35e-d4da-46d3-adaf-93b7b6d36403)

Next we import our datasets from the Syn-Aesthetic repository for ease of access. Initially I kept them in Google Drive to work with at home. Looking forward to comparing the other available datasets. Initially this project was focused solely on good chills, with inspirational speeches, frisson in music, and good chills in movies being the focus. As the project has evolved the I am  interested in expanding the spotlight to driving and piloting simulations, aautonomous vehicles, and other deep reinforcement learning systems. 

![Screen Shot 2024-01-20 at 2 32 11 PM](https://github.com/JessemanGray/Syn-Aesthetics/assets/123507565/e83eb232-9c59-4848-9a78-19dad2e2b74b)


![Screen Shot 2024-01-20 at 2 32 45 PM](https://github.com/JessemanGray/Syn-Aesthetics/assets/123507565/bd1e5076-7514-4f23-aad4-0540cb74fb24)


As a visual learner, using eye candy is a core part of my EDA process. By pushing the initial data set through various filters and comparing multiple variabless, a different storyline than the one I had originally made an outline for emerged. As one of my first completed self-driven projects that I felt had value in sharing with the data science and machine learning community, the patterns that stood out inspired me to take a different direction entirely, and allowed some leaps of thought that wouldn't have been possible otherwise.


Another aspect that may seem obscure but I feel is worth noting: one of the principles that I feel should be built in to any successful project is factoring in its environmental impact. Even though we can quantify the significant value that simple fixes like having a dark background, or prioritizing red, green, and orange over blue seem pointless compared to the degree of climate change occuring, but there is a power in being able to exponentially increase the benefits, as opposed to deferring to the status quo.

It may seem like just optics 👀 if there are not other measures, but even then it can raise awareness of the need for micro-progessions as well as major improvements in consumption habits.

By tweaking the basic tools used to complete common visualization tasks, the data is not only more appealing to my personal sensibilities, but has a distinctly smaller footprint than many Power BI, Tableau, and Excel dashboards. When we hard code vectorization and quantization into the model, we get an even better performance. As someone who's main experience with sustainable design is in building and bio-char, I am still learning about best practices in data managemant, storage, and development, so open to suggestions and contributions:)


![Screen Shot 2024-01-20 at 2 36 07 PM](https://github.com/JessemanGray/Syn-Aesthetics/assets/123507565/c27c4c5c-f386-469a-bdf0-3f950a9eba14)


Here is a rabbit-hole that I would like to explore related to the data viz of this project. By removing all values and subject matter we get an approximation of the data points that resembles a unique cloud structure. By abstracting the relationships we can make inferences that might not be as apparent as labeled data. This basic principle could have legs in regards to computer vision and re-inputting to see how a GAN model might interpret it. 

![Screen Shot 2024-01-20 at 2 37 47 PM](https://github.com/JessemanGray/Syn-Aesthetics/assets/123507565/507e8d3d-7caf-4e9f-b5c9-9b8ed7fde78a)

Training and Validation Loss and Learning Rate. 

![Screen Shot 2024-01-20 at 2 38 17 PM](https://github.com/JessemanGray/Syn-Aesthetics/assets/123507565/74e0770a-b2f3-4556-857f-6143a117be15)

These represent only a few of the visualizations and predictive models used in this project, and only a fraction of the ones I hope to usein future exploratory analysis. The best performing so far turned out to be an LSTM, but plan to employ Lasso, Elastic-net on the conservative side, and iForest and iTree as a bleeding edge example.

![Screen Shot 2024-01-20 at 2 38 25 PM](https://github.com/JessemanGray/Syn-Aesthetics/assets/123507565/7c262609-8f2d-4ad0-b27d-3418546706ca)

Usage:

This project is a preliminary exploration of this data that could eventually be used to accurately gauge biometric responses to various media. Possible applications could include productizing to be included in fitbit and other wearable tchnologies, and research into quantifying the uncanny valleyt effect in different population groups. 

Eventually I hope to expand current notebooks to include EDA of other datasets in ChillsDB, and continue sifting through to perform more fine-grained analysis. I also plan to include ongoing research into other examples of the uncanny valley effect in popular culture and historic context, such as reigious likenesses and artifacts, and the trope of the zombie apocalypse. By dissecting the concepts of 'good bad', and bad good', along with what I call the 'stained glass' effect, which is the ehereal quality of light that many GAN image generators seem to understand implicitly. 

License:

This project is licensed under the MIT License - see the LICENSE.md file for details

Contributions:

Any and all insight into the phenomenon of the uncanny valley is hughly encouraged.
If so inclined, please follow the following guidelines:

please 🍴 with this repository

grow a new 🌱 to keep changes organized

💍 please write clear commit messages

push to the 🍴 repository to upload your changes to GitHub

please submit a pull request so we can review your changes 🔍

Your time and contribution is valued ⏳ 💰

Thanks!


Contact:

Jesse Gray
jessemangray@gmail.com
jessemangray@icloud.com

