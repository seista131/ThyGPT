# ThyGPT

# Multimodal ChatGPT-Style Model for Assisting Thyroid Nodule Diagnosis and Management: A Multicenter Diagnostic Study

*Abstract*

Using artificial intelligence (AI) models to analyze ultrasound (US) images offers a promising approach to assessing the risk of thyroid nodules. However, traditional AI models lack transparency and interpretability, undermining radiologists' trust in Al-based diagnoses. Herein, we developed a multimodal ChatGPT-style model called the generative pre-trained transformer for thyroid nodules (ThyGPT), aimed at connecting and aligning the semantics of US images and descriptive reports of nodules, to provide a transparent and interpretable AI copilot model for the risk assessment and management of thyroid nodules. US data from 64,133 patients across nine hospitals, including 534,795 US images and 53,721 US examination reports, were retrospectively collected to train and test the model. After training, ThyGPT was found to assist in reducing biopsy rates by over 40% without increasing the rate of missed diagnoses. Meanwhile, it can detect errors in US reports 1,596 times faster than humans. With the assistance of ThyGPT, the area under the curve for radiologists in assessing the risk of thyroid nodules improved from 0.795 (95% confidence interval [ CI): 0.789-0.801) to 0.927 (95% Cl: 0.923-0.930; p < 0.001). As an Al-generated content-enhanced computer-aided diagnosis (AIGC-CAD) model, ThyGPT shows promise as a new-generation thyroid nodule CAD system, potentially revolutionizing how radiologists utilize CAD.

*Introduction:*

Thyroid nodules are a common endocrine condition, with a prevalence of over 60% in adults and an incidence three times higher in women than in men. Most of these nodules are benign, while only about
7-15% are malignant. In clinical practice, ultrasound (US) imaging and fine-needle aspiration (FNA)
biopsy are the primary methods used to assess the risk of thyroid nodules. However, the diagnostic outcomes of ultrasonography rely heavily on radiologists' experiences and skills. Even with FNA, precise risk evaluation remains elusive for over 15% of nodules. To a certain extent, these uncertainties in the risk assessment of thyroid nodules have led to a widespread issue of overdiagnosis and overtreatment, such as unnecessary biopsies or invasive surgeries for nodules ultimately determined to be benign8-10. This phenomenon not only inflicts significant physical and psychological trauma upon patients but also substantially augments healthcare expenditure. The current situation of overtreatment of thyroid nodules emphasizes the necessity for more refined and accurate risk assessment tools.
Recently, studies have indicated that computer-aided diagnosis (CAD) based on US images and artificial intelligence (AI) models holds promise as an effective complementary solution. In general, these studies have focused on developing specialized AI models, such as ThyNet, RedImageNet, and DeepThy-Net, to extract latent features from large US image data sets and assess the risk of thyroid nodules. These US image-based CAD models have made substantial progress; however, they also have significant shortcomings. First, existing CAD models lack transparency and cannot provide the rationale or decision-making basis behind their diagnoses, creating a gap in understanding between radiologists and CAD models. This "black box"characteristic undermines the confidence of radiologists, patients, and healthcare administrators in the diagnostic results of these CAD models. Second, the outputs of existing CAD models are mostly simple probability values or categorical labels; therefore, they lack meaningful interaction with radiologists. This "mute box"characteristic renders challenges for radiologists indistinguishing which AI-based diagnoses are accurate and reasonable and which may be erroneous or Alhallucinations. These barriers in communication and comprehension have even led many radiologists to abandon these AI-based CAD methods



#Project Overview
ThyroidGPT is an AI-powered system designed for thyroid diagnosis and consultation, utilizing a large language model such as LLaMA. The system integrates advanced natural language processing (NLP) techniques to assist healthcare professionals in diagnosing thyroid-related conditions. It also incorporates external knowledge bases and vector stores to enhance the accuracy and relevance of its responses.

*Large Language Model Directory*
This directory contains the core code of the large language model that powers ThyroidGPT. It includes the main logic for natural language understanding, processing, and generation.

*image Directory*
This directory is dedicated to pre-processing images and collecting multimodal data for detection. It supports the integration of visual information with the language model, enabling a more comprehensive analysis and diagnosis.

*AIGC_evaluation and Other Directories*
These directories are used to evaluate the performance of the ThyroidGPT model and compare it with other large language models. They include benchmarking scripts, evaluation metrics, and results analysis tools that help in assessing the model's accuracy, efficiency, and overall effectiveness.
