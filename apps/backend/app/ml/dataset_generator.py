"""Synthetic dataset generator for CareerFit ML.

Generates a balanced, realistic, domain-diverse dataset of labelled resume–job pairs
for training and evaluating machine learning models.
Zero PII: all resumes and job postings are strictly synthetic.
"""

from dataclasses import dataclass
import random
from typing import Optional
import pandas as pd


@dataclass
class DatasetExample:
    resume_text: str
    job_description: str
    label: int  # 1 = match, 0 = mismatch
    category: str


def build_curated_dataset() -> list[DatasetExample]:
    """Generate 360 unique, high-quality, realistic resume-job pairs across 8 domains."""
    examples: list[DatasetExample] = []

    # -------------------------------------------------------------------------
    # DOMAIN 1: Artificial Intelligence and Machine Learning (45 examples)
    # -------------------------------------------------------------------------
    ai_positives = [
        (
            "Machine Learning Engineer with 3 years of industry experience developing and deploying deep learning models. Proficient in Python, PyTorch, Scikit-Learn, and Hugging Face Transformers. Built automated text classification and semantic search pipelines utilizing BERT embeddings and Faiss vector search. Experienced with Docker, MLflow for experiment tracking, and deploying RESTful inference endpoints with FastAPI on AWS.",
            "Seeking an AI / Machine Learning Engineer to design, train, and deploy natural language processing models. Must have solid proficiency in Python, PyTorch or TensorFlow, Hugging Face, and Scikit-Learn. Experience with vector search, Docker, model deployment via REST APIs, and cloud services (AWS or GCP) is required.",
        ),
        (
            "Recent M.S. in Computer Science graduate specializing in Machine Learning and Computer Vision. Academic projects include building YOLOv8 object detection pipelines with OpenCV and PyTorch. Solid foundation in linear algebra, gradient descent optimization, NumPy, Pandas, and Python. Published thesis on self-supervised representation learning.",
            "Looking for a Junior Machine Learning Engineer to work on computer vision and deep learning applications. Requirements: Bachelor's or Master's in CS or related field, hands-on experience with Python, PyTorch, OpenCV, NumPy, and basic model evaluation techniques. Familiarity with convolutional neural networks is required.",
        ),
        (
            "Senior Deep Learning Scientist with 6 years experience architecting generative models and multimodal architectures. Expert in PyTorch, CUDA kernel profiling, distributed training with DeepSpeed, and PyTorch Lightning. Led a team optimizing transformer inference latency with TensorRT and ONNX runtime, achieving 40% speedup on GPU clusters.",
            "We are hiring a Senior ML / Deep Learning Scientist to drive generative AI initiatives. Qualifications: Extensive background in PyTorch, distributed training (DeepSpeed / Horovod), transformer architectures, and model quantization/optimization (TensorRT, ONNX). Track record of shipping production ML solutions.",
        ),
        (
            "AI Research Engineer with strong background in reinforcement learning, PyTorch, JAX, and Gymnasium. Developed custom policy gradient algorithms and simulated robotic control environments. Proficient in Python, Pandas, statistical modeling, and scientific computing.",
            "Seeking an AI Research Engineer with expertise in Reinforcement Learning and deep learning frameworks (PyTorch or JAX). Must have strong mathematical fundamentals, experience with simulation environments, and proficiency in Python.",
        ),
        (
            "Machine Learning Operations (MLOps) Engineer with 4 years experience. Expertise in automating end-to-end ML lifecycles using Kubeflow, MLflow, Docker, and Kubernetes on GCP. Maintained model monitoring dashboards for feature drift detection and automated model retraining pipelines with Python and Scikit-Learn.",
            "Hiring an MLOps Engineer to build scalable machine learning infrastructure. Must have strong skills in Python, Docker, Kubernetes, Kubeflow, CI/CD for ML, and experiment tracking with MLflow or Weights & Biases.",
        ),
        (
            "Computer Vision Specialist with 4 years hands-on experience in image segmentation, 3D point cloud processing, and facial recognition systems using PyTorch, OpenCV, and C++. Built edge-AI models optimized for NVIDIA Jetson devices.",
            "Looking for a Computer Vision Engineer to build real-time visual recognition systems. Requirements: Strong Python and C++ skills, deep understanding of OpenCV, CNN architectures, PyTorch or TensorFlow, and experience deploying to edge devices.",
        ),
        (
            "NLP Engineer with 3 years experience building conversational agents, LLM prompt pipelines, and retrieval-augmented generation (RAG) systems using LangChain, LlamaIndex, OpenAI API, and ChromaDB. Proficient in Python, spaCy, and FastAPI.",
            "Seeking an NLP Engineer to build enterprise generative AI and search solutions. Must have experience with Python, LangChain, RAG architecture, vector databases, and prompt engineering.",
        ),
        (
            "Data Science and Machine Learning Fresher with B.Tech in Artificial Intelligence. Completed capstone project predicting housing prices using XGBoost and Random Forest in Scikit-Learn. Proficient in Python, Pandas, NumPy, Matplotlib, and SQL.",
            "Looking for an Entry-Level ML Associate to assist the data science team. Requirements: Degree in CS/AI/Data Science, good grasp of Python, Scikit-Learn, regression/classification algorithms, and exploratory data analysis using Pandas and SQL.",
        ),
        (
            "Applied Scientist with 5 years experience in recommendation systems and ranking algorithms. Expertise in collaborative filtering, matrix factorization, and two-tower neural networks using TensorFlow, Python, BigQuery, and Spark.",
            "Hiring an Applied ML Scientist for our personalization and recommendation team. Requires experience with TensorFlow or PyTorch, recommendation algorithms, large-scale data processing with Spark, and SQL.",
        ),
        (
            "AI Software Developer proficient in Python, PyTorch, FastAPI, Celery, and Redis. Built microservices for automated OCR text extraction and document summarization using transformer models. Experienced with Git, Pytest, and Linux environments.",
            "Seeking an AI Application Developer to integrate machine learning models into microservice backends. Required: Python, FastAPI or Django, experience consuming and serving ML models, Pytest, and Docker.",
        ),
        (
            "Speech Processing and Audio ML Engineer with 3 years experience. Trained automatic speech recognition (ASR) and text-to-speech (TTS) models using PyTorch, Torchaudio, and Kaldi. Strong knowledge of digital signal processing.",
            "Audio & Speech ML Engineer needed to develop voice AI technology. Requires experience in audio signal processing, PyTorch, ASR architectures (Whisper / Conformer), and Python.",
        ),
        (
            "Senior Machine Learning Engineer with 7 years experience in tabular modeling, feature stores (Feast), and high-frequency risk scoring using LightGBM, CatBoost, Scikit-Learn, and AWS SageMaker. Led ML governance and bias auditing.",
            "Seeking Lead ML Engineer for fintech fraud detection. Must have deep expertise in gradient boosting (LightGBM, XGBoost), feature engineering, AWS SageMaker, Scikit-Learn, and model explainability (SHAP).",
        ),
        (
            "Graduate Student in Computational Linguistics with strong Python, NLTK, spaCy, and PyTorch skills. Built named entity recognition (NER) pipelines and sentiment analysis models during academic internship.",
            "Junior NLP Researcher wanted. Qualifications: Proficiency in Python, natural language processing libraries (spaCy, Hugging Face, NLTK), and solid understanding of tokenization, transformers, and sequence classification.",
        ),
        (
            "Edge AI Developer experienced with TensorFlow Lite, ONNX runtime, and ARM Cortex microcontrollers. Converted float32 deep neural networks to int8 quantized formats for IoT sensors. Proficient in C++ and Python.",
            "Hiring Embedded / Edge AI Engineer. Requirements: Experience with TensorFlow Lite or TensorRT, model quantization, Python, and C/C++ embedded programming.",
        ),
        (
            "Machine Learning Practitioner with 2 years experience developing predictive maintenance algorithms for industrial IoT. Proficient in time-series forecasting using ARIMA, Prophet, and LSTM networks in Python.",
            "Predictive Maintenance ML Engineer needed. Must have experience with time-series modeling, Python (Pandas, Scikit-Learn, Keras/PyTorch), and sensor anomaly detection.",
        ),
        (
            "AI Engineer with focus on Knowledge Graphs and Semantic Reasoning. Built graph neural networks (GNNs) using PyTorch Geometric and Neo4j for biomedical link prediction. Strong Python and Cypher query skills.",
            "Seeking Knowledge Graph & AI Specialist. Required: Python, Graph Neural Networks (PyTorch Geometric or DGL), graph databases (Neo4j), and semantic data modeling.",
        ),
        (
            "Deep Learning Engineer skilled in generative adversarial networks (GANs) and diffusion models for synthetic image generation. Proficient in PyTorch, CUDA, WandB, and Linux shell scripting.",
            "We are looking for a Deep Learning Engineer focused on generative image synthesis. Must have proven experience with diffusion models, GANs, PyTorch, and multi-GPU training setups.",
        ),
        (
            "Data Science & AI Intern with hands-on coursework in supervised learning, unsupervised clustering, and neural networks. Built an end-to-end churn prediction model in Python using Scikit-Learn and Streamlit.",
            "Intern / Junior AI Engineer position. Candidates must demonstrate proficiency in Python, core machine learning algorithms, Scikit-Learn, and ability to build quick prototypes using Streamlit or Gradio.",
        ),
        (
            "Senior NLP Architect with 8 years experience leading conversational AI platforms. Architected multilingual semantic search supporting 15 languages using sentence transformers, Milvus vector database, and Kubernetes.",
            "NLP Solutions Architect needed to lead enterprise search modernization. Requires 7+ years in NLP, vector search architectures, transformer embeddings, Python, and cloud infrastructure.",
        ),
        (
            "Machine Learning Engineer specializing in AutoML and Hyperparameter Optimization. Experienced with Optuna, Ray Tune, Scikit-Learn, and Python for automated model tuning in production pipelines.",
            "Looking for an ML Engineer to enhance model optimization pipelines. Strong background in Python, Optuna / Hyperopt, Scikit-Learn, and automated ML workflows required.",
        ),
        (
            "Computer Vision & Deep Learning Intern with experience implementing ResNet and Vision Transformer architectures on custom image datasets using PyTorch and Albumentations.",
            "Entry-level Computer Vision Engineer to assist in image annotation, data augmentation, and model benchmarking using PyTorch and OpenCV.",
        ),
        (
            "AI Model Validation and Quality Engineer with 3 years experience testing ML models for fairness, adversarial robustness, and drift. Proficient in Python, Scikit-Learn, Great Expectations, and Evidently AI.",
            "Model Risk & AI Validation Analyst needed. Responsibilities include evaluating ML model metrics, testing boundary cases, monitoring data drift, and ensuring compliance using Python testing tools.",
        ),
        (
            "Senior ML Infrastructure Engineer with 5 years experience scaling distributed model training clusters using Slurm, Kubernetes, and Ray. Proficient in Python, PyTorch, and high-performance network storage.",
            "Seeking ML Infrastructure Engineer to manage high-performance compute clusters for deep learning research. Experience with Kubernetes, Ray, Slurm, and PyTorch required.",
        ),
    ]

    ai_negatives = [
        # Subtle near-misses (e.g. BI/SQL analyst or software dev without ML applying for senior AI role)
        (
            "Business Intelligence Analyst with 4 years experience creating executive dashboards in Tableau and writing SQL queries in PostgreSQL. Skilled in Excel pivot tables and reporting business metrics.",
            "Seeking an AI / Machine Learning Engineer to design, train, and deploy natural language processing models. Must have solid proficiency in Python, PyTorch or TensorFlow, Hugging Face, and Scikit-Learn.",
        ),
        (
            "Senior Frontend Engineer with 6 years experience in Vue.js, HTML5, CSS3, and Nuxt.js. Built complex web user interfaces and responsive mobile-first websites.",
            "Looking for an Applied ML Scientist to build statistical NLP and machine learning pipelines. Requirements include proficiency in Python, PyTorch/TensorFlow, Scikit-Learn, and vector search.",
        ),
        (
            "Network Administrator with 5 years experience configuring Cisco routers, firewalls, VLANs, and VPN gateways. Certified CCNA with strong troubleshooting skills.",
            "We are hiring a Senior ML / Deep Learning Scientist to drive generative AI initiatives. Qualifications: Extensive background in PyTorch, distributed training, transformer architectures, and CUDA.",
        ),
        (
            "Manual QA Tester with 3 years experience writing test cases, logging bug tickets in Jira, and performing exploratory testing on mobile applications.",
            "Seeking an AI Research Engineer with expertise in Reinforcement Learning and deep learning frameworks (PyTorch or JAX). Must have strong mathematical fundamentals.",
        ),
        (
            "WordPress Developer with 4 years experience building marketing websites, customizing PHP themes, and managing WooCommerce plugins.",
            "Hiring an MLOps Engineer to build scalable machine learning infrastructure. Must have strong skills in Python, Docker, Kubernetes, Kubeflow, CI/CD for ML, and MLflow.",
        ),
        (
            "Executive Chef with 8 years culinary experience managing commercial kitchens, kitchen inventory, and menu creation.",
            "Looking for a Computer Vision Engineer to build real-time visual recognition systems. Requirements: Strong Python and C++ skills, OpenCV, and deep understanding of CNNs.",
        ),
        (
            "Graphic Designer with 5 years experience in Adobe Illustrator, Photoshop, and Figma creating brand logos, marketing brochures, and vector assets.",
            "Seeking an NLP Engineer to build enterprise generative AI and search solutions. Must have experience with Python, LangChain, RAG architecture, and vector databases.",
        ),
        (
            "High School Mathematics Teacher with 6 years experience teaching algebra, geometry, and calculus to secondary school students.",
            "Looking for an Entry-Level ML Associate to assist the data science team. Requirements: Degree in CS/AI, hands-on Python, Scikit-Learn, and regression/classification algorithms.",
        ),
        (
            "Customer Support Representative with 3 years experience answering helpdesk tickets, managing chat queues, and resolving billing issues.",
            "Hiring an Applied ML Scientist for our personalization and recommendation team. Requires experience with TensorFlow or PyTorch, recommendation algorithms, and Spark.",
        ),
        (
            "Real Estate Sales Agent with 7 years experience showing properties, closing escrow deals, and managing MLS listings.",
            "Seeking an AI Application Developer to integrate machine learning models into microservice backends. Required: Python, FastAPI or Django, ML model serving, and Docker.",
        ),
        (
            "Office Administrative Assistant skilled in calendar management, travel bookings, Microsoft Word, and office supply ordering.",
            "Audio & Speech ML Engineer needed to develop voice AI technology. Requires experience in audio signal processing, PyTorch, and ASR architectures (Whisper).",
        ),
        (
            "Civil Site Engineer with 5 years experience supervising concrete foundations, structural steel inspections, and CAD drawings.",
            "Seeking Lead ML Engineer for fintech fraud detection. Must have deep expertise in gradient boosting (LightGBM, XGBoost), feature engineering, and AWS SageMaker.",
        ),
        (
            "Content Writer and Copywriter with 4 years experience drafting SEO blog articles, social media captions, and email marketing newsletters.",
            "Junior NLP Researcher wanted. Qualifications: Proficiency in Python, natural language processing libraries (spaCy, Hugging Face), and transformer architectures.",
        ),
        (
            "Warehouse Logistics Supervisor with 6 years experience managing forklift operations, inventory cycle counts, and shipping manifests.",
            "Hiring Embedded / Edge AI Engineer. Requirements: Experience with TensorFlow Lite or TensorRT, model quantization, Python, and C/C++ embedded programming.",
        ),
        (
            "Licensed Massage Therapist with 5 years experience in deep tissue therapy, sports rehabilitation, and clinic reception.",
            "Predictive Maintenance ML Engineer needed. Must have experience with time-series modeling, Python (Pandas, Scikit-Learn, Keras/PyTorch), and sensor anomaly detection.",
        ),
        (
            "Travel Agent with 4 years experience booking airline tickets, hotel itineraries, and vacation packages using Amadeus GDS.",
            "Seeking Knowledge Graph & AI Specialist. Required: Python, Graph Neural Networks (PyTorch Geometric or DGL), and graph databases (Neo4j).",
        ),
        (
            "Automotive Mechanic with 8 years experience diagnosing engine transmissions, brake pad replacements, and vehicle electrical circuits.",
            "We are looking for a Deep Learning Engineer focused on generative image synthesis. Must have proven experience with diffusion models, GANs, and PyTorch.",
        ),
        (
            "Retail Store Cashier with 2 years experience handling POS transactions, cash register balances, and customer returns.",
            "Intern / Junior AI Engineer position. Candidates must demonstrate proficiency in Python, core machine learning algorithms, Scikit-Learn, and Streamlit.",
        ),
        (
            "Hotel Front Desk Clerk with 4 years hospitality experience checking in guests, managing room assignments, and handling concierge requests.",
            "NLP Solutions Architect needed to lead enterprise search modernization. Requires 7+ years in NLP, vector search architectures, transformer embeddings, and Python.",
        ),
        (
            "HVAC Technician with 6 years experience installing air conditioning units, refrigerant recharging, and furnace ductwork repair.",
            "Looking for an ML Engineer to enhance model optimization pipelines. Strong background in Python, Optuna / Hyperopt, Scikit-Learn, and automated ML workflows.",
        ),
        (
            "Fitness Trainer and Gym Instructor with 5 years experience creating personalized workout plans and leading group aerobic sessions.",
            "Entry-level Computer Vision Engineer to assist in image annotation, data augmentation, and model benchmarking using PyTorch and OpenCV.",
        ),
        (
            "Dental Hygienist with 4 years experience performing teeth cleanings, taking dental x-rays, and sterilizing medical instruments.",
            "Model Risk & AI Validation Analyst needed. Responsibilities include evaluating ML model metrics, testing boundary cases, and monitoring data drift using Python.",
        ),
    ]

    for r, j in ai_positives:
        examples.append(DatasetExample(r, j, 1, "AI & Machine Learning"))
    for r, j in ai_negatives:
        examples.append(DatasetExample(r, j, 0, "AI & Machine Learning"))

    # -------------------------------------------------------------------------
    # DOMAIN 2: Data Science and Data Analytics (45 examples)
    # -------------------------------------------------------------------------
    ds_positives = [
        (
            "Data Scientist with 3 years experience utilizing Python, SQL, Pandas, NumPy, and Scikit-Learn for exploratory data analysis, hypothesis testing, and predictive modeling. Built customer churn and lifetime value models that increased retention by 18%. Created interactive data visualizations using Seaborn, Plotly, and Tableau.",
            "We are seeking a Data Scientist to analyze complex datasets and develop statistical models. Requirements: Proficiency in Python, SQL, Pandas, Scikit-Learn, statistical hypothesis testing, and data visualization tools (Tableau or Power BI). Experience with predictive modeling is required.",
        ),
        (
            "Senior Quantitative Analyst with 6 years experience in financial modeling, risk analysis, and time-series econometrics using Python, R, and SQL. Developed Monte Carlo simulations and portfolio optimization algorithms. Proficient with Git, Jupyter, and PostgreSQL.",
            "Looking for a Senior Quantitative Data Scientist. Must have strong expertise in statistical modeling, Python or R, advanced SQL, time-series analysis, and financial risk forecasting.",
        ),
        (
            "Data Analyst with 2 years experience writing complex SQL queries, building executive dashboards in Power BI, and conducting exploratory analysis in Python (Pandas). Transformed raw ERP data into actionable KPIs for business stakeholders.",
            "Hiring a Data Analyst to translate business data into visual insights. Required: Strong SQL skills (joins, window functions), Power BI or Tableau dashboard development, and basic Python for data cleaning.",
        ),
        (
            "Product Data Analyst with 4 years experience designing and evaluating A/B tests, measuring user engagement funnels, and tracking product metrics using SQL, Mixpanel, Amplitude, and Python.",
            "Seeking a Product Analyst to partner with product managers. Must have hands-on experience designing A/B experiments, statistical significance testing, SQL, and product telemetry tools.",
        ),
        (
            "Recent Data Science Graduate with B.S. in Statistics. Completed projects in sentiment analysis and customer segmentation using K-Means and PCA in Scikit-Learn and R. Proficient in SQL, Excel, and Python.",
            "Entry-Level Data Scientist wanted. Requirements: Degree in Statistics, Mathematics, or Data Science; practical knowledge of Python (NumPy, Pandas, Scikit-Learn), basic SQL, and fundamentals of regression and clustering.",
        ),
        (
            "Marketing Data Scientist with 4 years experience modeling marketing attribution, customer acquisition cost, and marketing mix modeling (MMM) using Python, PyMC, SQL, and Google BigQuery.",
            "Looking for a Marketing Data Scientist. Requires experience with marketing attribution modeling, SQL, BigQuery, Python, and statistical inference.",
        ),
        (
            "Healthcare Data Analyst with 3 years experience analyzing clinical trial records, patient outcomes, and electronic health records (EHR). Proficient in SAS, SQL, Python, and HIPAA-compliant data handling.",
            "Hiring a Healthcare Data Analyst. Qualifications: Background in biostatistics or healthcare analytics, proficiency in SQL, Python or SAS, and familiarity with medical coding datasets.",
        ),
        (
            "Business Data Analyst skilled in Snowflake, dbt, SQL, and Tableau. Built automated ETL reporting views and financial metric models for revenue operations teams.",
            "Seeking a Business Intelligence Data Analyst. Must have strong skills in SQL, Snowflake, dbt data transformations, and Tableau dashboard design.",
        ),
        (
            "Senior Data Scientist specializing in experimental design, causal inference, and observational studies using DoWhy, Python, and SQL. Advised leadership on strategic pricing models.",
            "Lead Data Scientist wanted for economics and causal inference team. Strong knowledge of causal inference, econometrics, Python, and SQL required.",
        ),
        (
            "Data Science Intern with strong foundations in Python, Pandas, Matplotlib, and SQL. Built predictive classification models for credit default risk during university fellowship.",
            "Data Science Internship: Ideal candidate has strong academic knowledge of Python, Pandas, Scikit-Learn, SQL, and enthusiasm for data cleaning and visualization.",
        ),
        (
            "Supply Chain Data Analyst with 3 years experience forecasting inventory replenishment and warehouse logistics using Python, Excel VBA, and SQL. Reduced stockouts by 22%.",
            "Supply Chain Analyst needed. Requirements: Strong proficiency in SQL, Excel, Python for data manipulation, and experience analyzing inventory and logistics datasets.",
        ),
        (
            "Operations Data Analyst with 3 years experience tracking SLA metrics, call center efficiency, and workforce staffing using SQL, Google Sheets, and Looker.",
            "Operations Analyst position. Must be skilled in SQL, Looker or Tableau dashboard building, and operational metric analysis.",
        ),
        (
            "Data Analytics Consultant with 5 years experience delivering end-to-end analytics roadmaps for retail clients using SQL, Python, Databricks, and Power BI.",
            "Hiring an Analytics Consultant. Requires 4+ years in client-facing analytics, strong SQL, Databricks/Spark familiarity, and Power BI visualization expertise.",
        ),
        (
            "Fraud Data Analyst with 3 years experience detecting suspicious financial transactions using SQL, Python, and anomaly detection algorithms (Isolation Forest).",
            "Fraud Analytics Specialist wanted. Must have strong SQL, experience analyzing transactional log data, and familiarity with anomaly detection in Python.",
        ),
        (
            "Customer Insights Analyst skilled in surveying methodology, Net Promoter Score (NPS) analytics, and text mining customer reviews using Python (NLTK) and SQL.",
            "Seeking Customer Insights Analyst. Experience with customer sentiment analysis, text mining, SQL, and survey analytics tools required.",
        ),
        (
            "Junior Data Scientist with 1 year experience building regression and classification pipelines using Scikit-Learn, Pandas, and Flask for model demo interfaces.",
            "Junior Data Scientist to support predictive analytics initiatives. Requirements: Python, Pandas, Scikit-Learn, basic web deployment (Flask/Streamlit), and SQL.",
        ),
        (
            "Senior Analytics Engineer bridging data science and engineering with expertise in SQL, Python, Snowflake, dbt, and Great Expectations for data quality validation.",
            "Analytics Engineer needed to build reliable data transformation pipelines. Requires strong SQL, Python, dbt, Snowflake, and automated data testing experience.",
        ),
        (
            "Geospatial Data Scientist experienced in analyzing satellite imagery and GIS data using GeoPandas, Shapely, QGIS, Python, and PostGIS.",
            "Geospatial Data Analyst wanted. Must have experience with GIS software, PostGIS, GeoPandas, Python, and spatial data analytics.",
        ),
        (
            "HR Data Analyst with 3 years experience analyzing employee turnover, compensation benchmarking, and workforce diversity metrics using SQL, Excel, and Power BI.",
            "People Analytics Specialist. Requires experience in HR analytics, SQL, Excel advanced modeling, and Power BI dashboard creation.",
        ),
        (
            "E-commerce Data Analyst with 4 years experience optimizing conversion rates, cart abandonment funnels, and customer segmentation using Google Analytics 4, BigQuery, and SQL.",
            "E-commerce Growth Analyst needed. Must have strong SQL, Google Analytics 4, BigQuery, and conversion rate optimization (CRO) analytical skills.",
        ),
        (
            "Credit Risk Analyst with 4 years experience developing scorecard models, probability of default (PD) estimations, and regulatory capital calculations using SAS, Python, and SQL.",
            "Risk Modeling Analyst position. Qualifications: Experience with credit scoring models, logistic regression, SAS/Python, and SQL in banking environments.",
        ),
        (
            "Energy & Sustainability Data Analyst with experience analyzing smart meter sensor data and carbon footprint metrics using Python, Pandas, and SQL.",
            "Sustainability Data Analyst needed to build carbon accounting models. Strong Python, Pandas, and environmental metric modeling skills required.",
        ),
        (
            "Clinical Trial Statistician with M.S. in Biostatistics. Experienced in survival analysis (Kaplan-Meier, Cox Proportional Hazards) using R, SAS, and Python.",
            "Biostatistician wanted for clinical trial analysis. Requires proficiency in R or SAS, survival analysis, clinical protocol design, and statistical reporting.",
        ),
    ]

    ds_negatives = [
        (
            "Front Desk Concierge with 4 years experience greeting visitors, scheduling boardroom meetings, and sorting courier packages.",
            "We are seeking a Data Scientist to analyze complex datasets and develop statistical models. Requirements: Python, SQL, Pandas, Scikit-Learn, and Tableau.",
        ),
        (
            "Automotive Assembly Line Worker with 5 years experience inspecting welding seams and operating hydraulic press machinery.",
            "Looking for a Senior Quantitative Data Scientist. Must have strong expertise in statistical modeling, Python or R, advanced SQL, and financial risk forecasting.",
        ),
        (
            "Pastry Baker with 6 years experience baking artisan croissants, sourdough breads, and decorative wedding cakes.",
            "Hiring a Data Analyst to translate business data into visual insights. Required: Strong SQL skills, Power BI or Tableau, and basic Python.",
        ),
        (
            "Plumbing Contractor with 8 years experience installing copper pipe networks, water heater replacements, and drain snaking.",
            "Seeking a Product Analyst to partner with product managers. Must have hands-on experience designing A/B experiments, statistical significance testing, and SQL.",
        ),
        (
            "Elementary School Art Teacher with 5 years experience teaching watercolor painting, ceramics, and papier-mâché crafts.",
            "Entry-Level Data Scientist wanted. Requirements: Degree in Statistics/Math, practical knowledge of Python (NumPy, Pandas, Scikit-Learn), and basic SQL.",
        ),
        (
            "Commercial Truck Driver with clean CDL Class A license, 7 years long-haul freight experience, and electronic logbook management.",
            "Looking for a Marketing Data Scientist. Requires experience with marketing attribution modeling, SQL, BigQuery, Python, and statistical inference.",
        ),
        (
            "Hair Stylist and Salon Colorist with 6 years experience in precision scissor cuts, balayage coloring, and client consultations.",
            "Hiring a Healthcare Data Analyst. Qualifications: Background in biostatistics or healthcare analytics, proficiency in SQL, Python or SAS.",
        ),
        (
            "Forklift Operator with OSHA certification and 4 years experience loading pallets onto freight trailers in cold storage warehouses.",
            "Seeking a Business Intelligence Data Analyst. Must have strong skills in SQL, Snowflake, dbt data transformations, and Tableau.",
        ),
        (
            "Security Guard with 3 years experience monitoring CCTV cameras, logging building visitors, and conducting perimeter patrols.",
            "Lead Data Scientist wanted for economics and causal inference team. Strong knowledge of causal inference, econometrics, Python, and SQL required.",
        ),
        (
            "Bartender with 5 years experience mixing craft cocktails, managing bar cash registers, and inventorying liquor supplies in high-volume venues.",
            "Data Science Internship: Ideal candidate has strong academic knowledge of Python, Pandas, Scikit-Learn, SQL, and enthusiasm for data cleaning.",
        ),
        (
            "Landscape Gardener with 6 years experience operating riding lawnmowers, trimming hedges, and installing irrigation drip lines.",
            "Supply Chain Analyst needed. Requirements: Strong proficiency in SQL, Excel, Python for data manipulation, and experience analyzing inventory.",
        ),
        (
            "Roofing Installer with 7 years experience laying asphalt shingles, flashing chimney borders, and repairing roof leaks.",
            "Operations Analyst position. Must be skilled in SQL, Looker or Tableau dashboard building, and operational metric analysis.",
        ),
        (
            "Flight Attendant with 5 years experience delivering in-flight passenger service, safety demonstrations, and emergency evacuations.",
            "Hiring an Analytics Consultant. Requires 4+ years in client-facing analytics, strong SQL, Databricks/Spark familiarity, and Power BI.",
        ),
        (
            "Dog Groomer and Pet Care Attendant with 4 years experience bathing, brushing, and styling canines of various breeds.",
            "Fraud Analytics Specialist wanted. Must have strong SQL, experience analyzing transactional log data, and anomaly detection in Python.",
        ),
        (
            "Drywall Finisher with 6 years experience taping, mudding, and sanding sheetrock in commercial renovation projects.",
            "Seeking Customer Insights Analyst. Experience with customer sentiment analysis, text mining, SQL, and survey analytics tools required.",
        ),
        (
            "Florist and Floral Designer with 4 years experience arranging seasonal flower bouquets, centerpieces, and managing fresh floral stock.",
            "Junior Data Scientist to support predictive analytics initiatives. Requirements: Python, Pandas, Scikit-Learn, basic web deployment, and SQL.",
        ),
        (
            "Janitorial Custodian with 5 years experience cleaning office buildings, buffing vinyl floors, and handling waste disposal.",
            "Analytics Engineer needed to build reliable data transformation pipelines. Requires strong SQL, Python, dbt, Snowflake, and data testing.",
        ),
        (
            "Auto Body Painter with 8 years experience prepping automotive metal surfaces, mixing polyurethane paints, and operating spray booths.",
            "Geospatial Data Analyst wanted. Must have experience with GIS software, PostGIS, GeoPandas, Python, and spatial data analytics.",
        ),
        (
            "Barista with 3 years experience preparing espresso beverages, latte art, and handling point-of-sale transactions in a busy coffee shop.",
            "People Analytics Specialist. Requires experience in HR analytics, SQL, Excel advanced modeling, and Power BI dashboard creation.",
        ),
        (
            "Carpenter with 7 years experience building custom wood cabinets, installing interior trim, and operating table saws.",
            "E-commerce Growth Analyst needed. Must have strong SQL, Google Analytics 4, BigQuery, and conversion rate optimization skills.",
        ),
        (
            "Scaffolding Erector with 4 years experience assembling galvanized steel scaffold towers on high-rise construction sites.",
            "Risk Modeling Analyst position. Qualifications: Experience with credit scoring models, logistic regression, SAS/Python, and SQL.",
        ),
        (
            "Swimming Pool Cleaner with 3 years experience balancing chlorine chemical levels, vacuuming pool tiles, and cleaning filter pumps.",
            "Sustainability Data Analyst needed to build carbon accounting models. Strong Python, Pandas, and environmental metric modeling skills required.",
        ),
    ]

    for r, j in ds_positives:
        examples.append(DatasetExample(r, j, 1, "Data Science & Analytics"))
    for r, j in ds_negatives:
        examples.append(DatasetExample(r, j, 0, "Data Science & Analytics"))

    # -------------------------------------------------------------------------
    # DOMAIN 3: Frontend and Backend Web Development (45 examples)
    # -------------------------------------------------------------------------
    web_positives = [
        (
            "Frontend Engineer with 4 years experience building responsive, accessible web applications using React, TypeScript, Next.js, and Tailwind CSS. Proficient in state management with Zustand and TanStack Query. Implemented comprehensive unit and integration tests using Vitest and React Testing Library. Strong focus on Core Web Vitals and performance optimization.",
            "We are seeking a Frontend React Engineer to build high-performance user interfaces. Required skills: Strong proficiency in React, TypeScript, Next.js, HTML5/CSS3, and modern component testing. Experience with state management and responsive design systems is essential.",
        ),
        (
            "Backend Software Developer with 5 years experience architecting scalable RESTful microservices and event-driven backends using Node.js, Express, TypeScript, PostgreSQL, and Redis. Experienced with Docker, message queues (RabbitMQ), and automated CI/CD deployment pipelines on AWS.",
            "Hiring a Backend Node.js Engineer. Key requirements: Extensive experience with Node.js, TypeScript, Express or NestJS, relational databases (PostgreSQL), Redis caching, and microservice architecture. Docker and AWS familiarity required.",
        ),
        (
            "Full Stack Developer proficient in the MERN stack (MongoDB, Express, React, Node.js) with 3 years experience. Built end-to-end web applications with JWT authentication, Stripe payment integrations, and responsive UI components. Familiar with Git and agile workflows.",
            "Looking for a Full Stack JavaScript / TypeScript Developer. Must have hands-on experience with React on the frontend and Node.js/Express on the backend, alongside NoSQL or SQL databases.",
        ),
        (
            "Junior Frontend Developer with strong fundamentals in JavaScript (ES6+), React, HTML5, CSS Grid, and responsive web design. Completed bootcamp building modern single-page applications connected to REST APIs. Experienced with Git version control.",
            "Seeking a Junior Frontend Web Developer to assist in building modern web portals. Requirements: Solid knowledge of JavaScript, React, HTML, CSS, Git, and ability to consume RESTful endpoints.",
        ),
        (
            "Senior Backend Engineer with 7 years experience in Golang, gRPC, and distributed microservices. Designed high-concurrency payment gateways processing 15k transactions per second with low latency. Strong background in PostgreSQL, Kafka, and Kubernetes.",
            "We are hiring a Senior Go / Backend Infrastructure Engineer. Must have deep experience with Golang, gRPC, distributed systems, PostgreSQL, Kafka, and containerized deployment in Kubernetes.",
        ),
        (
            "UI/UX Frontend Developer with 4 years experience bridging Figma design and code using Vue.js, Pinia, TypeScript, and Tailwind CSS. Dedicated to web accessibility standards (WCAG 2.1 AA) and cross-browser consistency.",
            "Frontend UI Developer needed. Must be proficient in modern frontend frameworks (Vue.js or React), TypeScript, translating Figma specs into pixel-perfect CSS, and accessibility best practices.",
        ),
        (
            "Backend Java Developer with 4 years experience building enterprise backends using Java 17, Spring Boot, Spring Data JPA, and PostgreSQL. Experienced in designing REST APIs, JUnit testing, and Maven build automation.",
            "Seeking Java Spring Boot Backend Developer. Requirements: Strong experience with Java, Spring Boot, microservices architecture, Hibernate/JPA, relational databases, and unit testing.",
        ),
        (
            "Frontend Developer specializing in Angular, RxJS, TypeScript, and SCSS with 3 years experience developing enterprise portals. Experienced with karma testing, modular architecture, and HTTP interceptors.",
            "Looking for an Angular Web Developer to maintain and develop large-scale web applications. Strong proficiency in Angular, TypeScript, RxJS, and clean component architecture required.",
        ),
        (
            "Full Stack Web Engineer with experience in Python (FastAPI/Django) and React. Architected secure user authentication, database migrations with Alembic, and WebSocket real-time chat features. Deployed via Docker on Heroku and AWS.",
            "Full Stack Engineer needed. Requires experience with Python backend frameworks (FastAPI/Django), React frontend, SQL databases, and containerized deployments.",
        ),
        (
            "Web Performance & Core Web Vitals Specialist with 5 years experience optimizing Next.js and React web apps. Reduced Largest Contentful Paint (LCP) by 45% through code splitting, edge caching, and server-side rendering (SSR).",
            "Senior Web Performance Engineer. Strong background in Next.js, React SSR, web bundling tools (Webpack/Vite), edge CDNs, and browser rendering lifecycle required.",
        ),
        (
            "Backend API Engineer with 3 years experience building GraphQL and REST APIs with Node.js, Apollo Server, Prisma ORM, and PostgreSQL. Implemented rate limiting, OAuth2 login, and automated OpenAPI documentation.",
            "Hiring a Backend API Developer. Key qualifications: Node.js, GraphQL / REST API design, ORM tools (Prisma or TypeORM), and PostgreSQL database modeling.",
        ),
        (
            "Entry-Level Web Developer with Computer Science degree. Built full-stack e-commerce portfolio site using React, Node.js, Express, and MongoDB. Strong grasp of asynchronous JavaScript, DOM manipulation, and CSS flexbox.",
            "Entry-Level Web Developer wanted. Ideal candidate has practical portfolio projects in modern JavaScript, HTML/CSS, React, and understanding of backend API connectivity.",
        ),
        (
            "Senior Frontend Architect with 8 years experience leading engineering teams building micro-frontend architectures with Module Federation, React, TypeScript, and Design Systems.",
            "Lead Frontend Architect to spearhead our UI modernization. Requires 7+ years in frontend development, design system governance, TypeScript, and micro-frontend patterns.",
        ),
        (
            "Backend Developer with 4 years experience in Ruby on Rails, PostgreSQL, Sidekiq background jobs, and RSpec testing. Built multi-tenant SaaS applications with stripe subscriptions.",
            "Ruby on Rails Backend Developer needed to scale our SaaS product. Requirements: Ruby on Rails, PostgreSQL, Redis/Sidekiq, and strong automated testing skills.",
        ),
        (
            "Full Stack TypeScript Engineer with 3 years experience using Remix, Next.js, Prisma, Tailwind CSS, and PlanetScale MySQL. Passionate about end-to-end type safety and clean UI design.",
            "Full Stack Developer wanted. Must have strong TypeScript proficiency across the full stack, Next.js or Remix experience, and familiarity with modern SQL databases.",
        ),
        (
            "Mobile-Responsive Web Developer with 3 years experience crafting cross-device layouts using CSS Grid, Flexbox, SASS, and vanilla JavaScript. Strong understanding of SEO fundamentals.",
            "Web Developer position focused on responsive marketing pages and landing pages. Requirements: HTML5, CSS3, modern JavaScript, responsive design, and basic CMS integration.",
        ),
        (
            "Backend Engineer with 3 years experience in C# and .NET Core. Developed enterprise REST APIs, Entity Framework Core repositories, and SQL Server databases in Azure environments.",
            "C# / .NET Backend Developer needed. Required: .NET Core 6+, C#, Entity Framework, SQL Server, and RESTful API architecture.",
        ),
        (
            "Frontend Engineer experienced in React, Redux Toolkit, WebSockets, and Canvas API for building interactive browser games and financial charting widgets.",
            "React Frontend Developer wanted for interactive data dashboard product. Requires React, Redux, WebSocket real-time feeds, and TypeScript.",
        ),
        (
            "Backend Microservices Developer with 4 years experience in PHP 8, Laravel, MySQL, Docker, and Redis. Implemented queue workers and automated PHPUnit test suites.",
            "Laravel / PHP Backend Engineer needed to build and scale web APIs. Requirements: PHP 8, Laravel, relational databases, REST APIs, and unit testing.",
        ),
        (
            "Web Accessibility (a11y) Specialist with 4 years experience conducting screen-reader testing (NVDA/JAWS), keyboard navigation audits, and implementing ARIA attributes in React.",
            "Accessibility Web Specialist. Must have strong understanding of WCAG standards, semantic HTML, ARIA, and practical experience remediating frontend React apps.",
        ),
        (
            "Junior Full Stack Developer with 1 year experience building CRUD applications in Express, React, and PostgreSQL. Familiar with Git, Postman API testing, and basic Linux commands.",
            "Junior Full Stack Developer to assist product engineering. Candidates should know JavaScript, React, basic Node.js, and SQL fundamentals.",
        ),
        (
            "Backend Platform Engineer with 5 years experience designing distributed caching layers, rate limiters, and authentication proxies using Go, Redis, and Envoy.",
            "Platform Backend Engineer. Strong background in systems architecture, Go, caching strategies, and high-volume API routing required.",
        ),
        (
            "Frontend Developer with 2 years experience building component libraries with Storybook, React, TypeScript, and Tailwind CSS. Focused on reusable modular code.",
            "Frontend Component Engineer to build and maintain shared UI library. Requirements: React, TypeScript, Storybook, and CSS utility frameworks.",
        ),
    ]

    web_negatives = [
        (
            "Commercial Electrician with 6 years experience wiring industrial switchboards, three-phase transformers, and conduit bending.",
            "We are seeking a Frontend React Engineer to build high-performance user interfaces. Required: React, TypeScript, Next.js, and CSS.",
        ),
        (
            "Heavy Equipment Operator with 8 years experience running bulldozers, excavators, and graders on highway grading projects.",
            "Hiring a Backend Node.js Engineer. Key requirements: Node.js, TypeScript, Express, PostgreSQL, Redis, and microservices.",
        ),
        (
            "Certified Dental Assistant with 4 years experience chairside assisting, dental impressions, and autoclave tool sterilization.",
            "Looking for a Full Stack JavaScript / TypeScript Developer. Must have hands-on experience with React on frontend and Node.js on backend.",
        ),
        (
            "Bricklayer and Mason with 7 years experience laying cinder blocks, brick veneers, and stone retaining walls.",
            "Seeking a Junior Frontend Web Developer. Requirements: Solid knowledge of JavaScript, React, HTML, CSS, and Git.",
        ),
        (
            "Flight Operations Coordinator with 5 years experience monitoring aircraft dispatch schedules and weather NOTAM briefings.",
            "We are hiring a Senior Go / Backend Infrastructure Engineer. Must have deep experience with Golang, gRPC, distributed systems, and PostgreSQL.",
        ),
        (
            "Veterinary Technician with 4 years experience administering animal vaccinations, surgical prep, and veterinary clinic reception.",
            "Frontend UI Developer needed. Must be proficient in modern frontend frameworks (Vue.js or React), TypeScript, and Figma CSS.",
        ),
        (
            "Commercial Painter with 5 years experience pressure washing exteriors, masking trim, and spraying epoxy coatings.",
            "Seeking Java Spring Boot Backend Developer. Requirements: Java, Spring Boot, microservices architecture, and PostgreSQL.",
        ),
        (
            "Hotel Housekeeping Supervisor with 6 years experience managing room attendants, linen supplies, and inspection checklists.",
            "Looking for an Angular Web Developer to maintain and develop large-scale web applications. Strong proficiency in Angular and TypeScript.",
        ),
        (
            "Automotive Detailer with 3 years experience performing paint clay bar decontamination, buffing scratches, and leather conditioning.",
            "Full Stack Engineer needed. Requires experience with Python backend frameworks (FastAPI/Django), React frontend, and SQL.",
        ),
        (
            "Security Patrol Officer with 4 years experience patrolling commercial strip malls and responding to security alarm calls.",
            "Senior Web Performance Engineer. Strong background in Next.js, React SSR, web bundling tools, and browser rendering lifecycle.",
        ),
        (
            "Tree Care Specialist and Arborist with 5 years experience climbing with rope harnesses, operating wood chippers, and pruning trees.",
            "Hiring a Backend API Developer. Key qualifications: Node.js, GraphQL / REST API design, ORM tools, and PostgreSQL.",
        ),
        (
            "Floral Arranger with 3 years experience creating sympathy wreaths, wedding table bouquets, and keeping cut flower coolers stocked.",
            "Entry-Level Web Developer wanted. Ideal candidate has practical portfolio projects in modern JavaScript, HTML/CSS, and React.",
        ),
        (
            "Bus Driver with 6 years experience operating public transit routes, passenger fare collection, and vehicle circle checks.",
            "Lead Frontend Architect to spearhead our UI modernization. Requires 7+ years in frontend development, TypeScript, and micro-frontends.",
        ),
        (
            "Locksmith with 5 years experience cutting master keys, rekeying commercial door cylinders, and installing deadbolts.",
            "Ruby on Rails Backend Developer needed. Requirements: Ruby on Rails, PostgreSQL, Redis/Sidekiq, and strong automated testing.",
        ),
        (
            "Meat Cutter and Butcher with 8 years experience breaking down beef sides, operating meat bandsaws, and grinding ground chuck.",
            "Full Stack Developer wanted. Must have strong TypeScript proficiency across the full stack, Next.js or Remix, and modern SQL.",
        ),
        (
            "Upholsterer with 6 years experience stripping antique chairs, replacing foam cushions, and sewing heavy vinyl fabric covers.",
            "Web Developer position focused on responsive marketing pages. Requirements: HTML5, CSS3, modern JavaScript, and responsive design.",
        ),
        (
            "Garbage Truck Driver with clean commercial driving record and 5 years residential automated arm waste collection experience.",
            "C# / .NET Backend Developer needed. Required: .NET Core 6+, C#, Entity Framework, SQL Server, and RESTful API architecture.",
        ),
        (
            "Shoe Repair Cobbler with 10 years experience resoling leather boots, stitching shoe soles, and stretching athletic footwear.",
            "React Frontend Developer wanted for interactive data dashboard product. Requires React, Redux, WebSocket feeds, and TypeScript.",
        ),
        (
            "Bicycle Mechanic with 4 years experience truing bicycle wheels, adjusting hydraulic disc brakes, and tuning derailleurs.",
            "Laravel / PHP Backend Engineer needed to build and scale web APIs. Requirements: PHP 8, Laravel, relational databases, and REST APIs.",
        ),
        (
            "Carpet Installer with 7 years experience power stretching broadloom carpets, seam heat welding, and tack strip installation.",
            "Accessibility Web Specialist. Must have strong understanding of WCAG standards, semantic HTML, ARIA, and React remediation.",
        ),
        (
            "Pest Control Technician with 4 years experience applying termite barrier treatments and setting rodent bait traps.",
            "Junior Full Stack Developer to assist product engineering. Candidates should know JavaScript, React, basic Node.js, and SQL.",
        ),
        (
            "Piano Tuner with 8 years experience tuning grand and upright acoustic pianos using acoustic tuning forks and electronic strobe meters.",
            "Platform Backend Engineer. Strong background in systems architecture, Go, caching strategies, and high-volume API routing.",
        ),
    ]

    for r, j in web_positives:
        examples.append(DatasetExample(r, j, 1, "Web Development"))
    for r, j in web_negatives:
        examples.append(DatasetExample(r, j, 0, "Web Development"))

    # -------------------------------------------------------------------------
    # DOMAIN 4: Python and Software Development (45 examples)
    # -------------------------------------------------------------------------
    python_positives = [
        (
            "Python Software Engineer with 4 years experience developing high-concurrency microservices, data extraction pipelines, and REST APIs using Python 3, FastAPI, Celery, and PostgreSQL. Proficient in Pytest, asynchronous programming (asyncio), Docker containerization, and Git. Strong background in data structures and object-oriented design patterns.",
            "Seeking a Python Developer to build and maintain backend microservices and workflow automation engines. Requirements: Strong proficiency in Python, FastAPI or Django, asyncio, relational databases (PostgreSQL), Redis, and automated unit testing with Pytest.",
        ),
        (
            "Senior Python Developer with 7 years experience in distributed architectures, multiprocessing, and high-throughput backend services using Django, Celery, Redis, and MySQL on AWS. Led database schema optimizations and architectural refactoring.",
            "Hiring a Senior Python Backend Architect. Must have 6+ years in Python, advanced knowledge of Django / FastAPI, asynchronous task processing with Celery, performance profiling, and cloud deployments.",
        ),
        (
            "Junior Python Programmer with B.S. in Computer Science. Built web scraping bots using BeautifulSoup and Playwright, and developed CRUD APIs with Flask and SQLite. Solid understanding of Python syntax, OOP concepts, list comprehensions, and Git.",
            "Looking for a Junior Python Developer. Requirements: Strong grasp of core Python, basic knowledge of Flask or Django, understanding of REST APIs, Git version control, and eagerness to learn.",
        ),
        (
            "Python Automation Engineer with 4 years experience creating end-to-end automated testing frameworks using Python, Pytest, Selenium, and Requests for REST API verification. Integrated test runs into Jenkins CI/CD pipelines.",
            "Seeking a Python Test Automation Engineer to build reliable automated verification suites. Requires solid Python scripting, Pytest, API testing, and CI/CD integration experience.",
        ),
        (
            "Python Data Engineering Specialist with 3 years experience building ETL data pipelines using Python, Pandas, Apache Airflow, and SQLAlchemy. Automated daily data extraction from external APIs into PostgreSQL data warehouses.",
            "Python Data Pipeline Developer needed. Qualifications: Hands-on Python, Airflow DAG development, SQLAlchemy or psycopg2, SQL data modeling, and automated error handling.",
        ),
        (
            "Systems Software Engineer proficient in Python and C++ for Linux systems programming. Developed IPC socket daemons and multi-threaded monitoring services. Strong knowledge of POSIX standards and memory management.",
            "Systems Software Engineer wanted. Experience with Python, C/C++, Linux socket programming, multithreading, and systems-level debugging required.",
        ),
        (
            "Python Developer specializing in Django Web Framework with 3 years experience. Built custom Django admin extensions, Django REST framework serializers, and managed database migrations with PostgreSQL.",
            "Django Developer needed for web portal development. Must have hands-on experience with Django, Django REST Framework, relational databases, and third-party API integrations.",
        ),
        (
            "Fintech Python Developer with 4 years experience building algorithmic trade execution connectors and market data feeds using asyncio, Websockets, and NumPy in Python. Implemented strict unit test coverage.",
            "Quantitative Python Software Engineer wanted for high-frequency trading platform. Strong skills in Python, asyncio, Websockets, numerical algorithms, and low-latency code required.",
        ),
        (
            "Python & FastApi Developer with 2 years experience building lightweight microservices, JWT token security, and Pydantic validation schemas. Deployed containerized applications with Docker Compose.",
            "FastAPI Developer needed to build modern microservice APIs. Requires Python, FastAPI, Pydantic, PostgreSQL, Docker, and Git.",
        ),
        (
            "Scientific Python Programmer with background in physics simulation and numerical analysis. Proficient in NumPy, SciPy, Matplotlib, and SymPy for mathematical algorithm implementation.",
            "Scientific Programmer wanted. Requires strong Python, NumPy, SciPy, algorithm implementation, and mathematical problem-solving skills.",
        ),
        (
            "Python Tools & Infrastructure Engineer with 3 years experience creating internal CLI tools using Click and Typer, managing packaging with Poetry, and automating developer workflows.",
            "Internal Tools Python Engineer. Experience building CLI utilities, Python package management (Poetry/Pipenv), and automated developer tooling required.",
        ),
        (
            "Entry-Level Software Engineer with strong Python coding fundamentals, solved 250+ LeetCode problems in data structures and algorithms, and built small open-source Python utility libraries.",
            "Associate Software Engineer. Ideal candidate has solid foundation in Python, algorithms, data structures, object-oriented programming, and collaborative Git experience.",
        ),
        (
            "Python Web Scraping Specialist with 3 years experience extracting high-volume web data using Scrapy, Selenium, and rotating proxies. Handled dynamic JavaScript rendering and anti-bot challenges.",
            "Web Data Extraction Engineer needed. Must have expertise in Python, Scrapy, parsing HTML/JSON, handling rate limits, and saving data to databases.",
        ),
        (
            "Full Stack Python Developer with experience in Flask backend and React frontend. Built database-driven dashboards, user authentication systems, and automated email notifications.",
            "Full Stack Python Engineer. Requires Python (Flask or FastAPI), React basics, SQL databases, and full lifecycle web development experience.",
        ),
        (
            "Embedded Linux Python Developer with 3 years experience scripting hardware diagnostic routines and sensor data readers on Raspberry Pi and BeagleBone boards using Python and I2C/SPI interfaces.",
            "Embedded Python Developer needed for IoT gateway development. Experience with Python on Linux SBCs, serial communication, and hardware interfacing required.",
        ),
        (
            "Python Cloud Function Specialist with 2 years experience writing serverless microservices on AWS Lambda and GCP Cloud Functions triggered by S3 events and SQS queues.",
            "Serverless Python Developer. Strong background in AWS Lambda, Python 3, API Gateway, event-driven architectures, and DynamoDB required.",
        ),
        (
            "Python QA / SDET with 4 years experience designing test harnesses for backend microservices using Pytest, Tavern, and Mock. Experienced with test-driven development (TDD).",
            "Software Development Engineer in Test (SDET - Python). Must have advanced Pytest experience, API testing skills, mock frameworks, and CI/CD automation.",
        ),
        (
            "Python GIS Developer experienced in building spatial analysis scripts using ArcPy, GDAL, and Shapely. Automated spatial vector transformations and map report generation.",
            "GIS Python Developer needed to build automated geospatial workflows. Requires Python, GDAL/OGR, ArcPy or QGIS scripting, and spatial databases.",
        ),
        (
            "Python Distributed Systems Engineer with 5 years experience building message-broker architectures using Celery, RabbitMQ, and Apache Kafka in Python.",
            "Distributed Systems Python Engineer. Deep knowledge of message queues, distributed task execution, Redis, and high-availability systems required.",
        ),
        (
            "Python Security Tooling Developer with 3 years experience building vulnerability scanners, log auditing scripts, and automated secrets detection tools.",
            "Security Automation Engineer. Strong Python scripting, familiarity with security scanning tools, and API integration capabilities required.",
        ),
        (
            "Python Game Backend Developer with 2 years experience building multiplayer matchmaking servers and inventory management APIs using asyncio and Redis.",
            "Game Backend Engineer (Python). Requires asyncio, Redis, WebSocket networking, and relational database management.",
        ),
        (
            "Python Desktop GUI Developer experienced in building cross-platform native utilities using PyQt6, PySide, and PyInstaller for Windows and macOS packaging.",
            "Desktop Application Developer wanted. Must be proficient in Python, PyQt or PySide, multi-threaded GUI design, and software bundling.",
        ),
        (
            "Junior Backend Python Developer with internship experience building CRUD REST endpoints with FastAPI and SQLAlchemy. Proficient in Git and Linux terminal.",
            "Junior Backend Developer. Candidates must demonstrate Python fundamentals, familiarity with relational databases, and RESTful API concepts.",
        ),
    ]

    python_negatives = [
        (
            "Hairdresser with 7 years salon experience specializing in highlights, blowouts, and hair extension fittings.",
            "Seeking a Python Developer to build and maintain backend microservices and workflow automation engines. Requirements: Python, FastAPI, and PostgreSQL.",
        ),
        (
            "House Painter with 5 years experience rolling interior latex paint, caulking baseboards, and scraping peeling trim.",
            "Hiring a Senior Python Backend Architect. Must have 6+ years in Python, advanced knowledge of Django / FastAPI, and Celery.",
        ),
        (
            "Baker with 4 years experience baking sourdough boules, bagels, and breakfast scones in an early morning commercial bakery.",
            "Looking for a Junior Python Developer. Requirements: Strong grasp of core Python, basic knowledge of Flask or Django, and Git.",
        ),
        (
            "Automotive Tire Technician with 3 years experience mounting truck tires, wheel balancing, and tire pressure sensor resets.",
            "Seeking a Python Test Automation Engineer to build reliable automated verification suites. Requires solid Python scripting and Pytest.",
        ),
        (
            "Dental Receptionist with 4 years experience checking in dental patients, verifying insurance eligibility, and collecting copays.",
            "Python Data Pipeline Developer needed. Qualifications: Hands-on Python, Airflow DAG development, SQLAlchemy, and SQL.",
        ),
        (
            "Drywall Hanger with 6 years experience measuring and screwing gypsum wallboards onto commercial metal studs.",
            "Systems Software Engineer wanted. Experience with Python, C/C++, Linux socket programming, and multithreading required.",
        ),
        (
            "Dog Daycare Attendant with 3 years experience supervising playgroups of 20+ dogs, feeding schedules, and kennel cleaning.",
            "Django Developer needed for web portal development. Must have hands-on experience with Django, Django REST Framework, and SQL.",
        ),
        (
            "School Bus Driver with 5 years experience transporting students safely, performing pre-trip vehicle checks, and maintaining order.",
            "Quantitative Python Software Engineer wanted for high-frequency trading platform. Strong skills in Python, asyncio, and low-latency code.",
        ),
        (
            "Bartender with 4 years experience mixing specialty drinks, opening wine bottles, and closing register drawers in a hotel lounge.",
            "FastAPI Developer needed to build modern microservice APIs. Requires Python, FastAPI, Pydantic, PostgreSQL, and Docker.",
        ),
        (
            "Plumbing Service Tech with 6 years experience clearing clogged kitchen sinks, repairing toilet fill valves, and installing garbage disposals.",
            "Scientific Programmer wanted. Requires strong Python, NumPy, SciPy, algorithm implementation, and mathematical problem-solving skills.",
        ),
        (
            "Landscape Laborer with 3 years experience spreading mulch, weeding flowerbeds, and trimming ornamental shrubs with hand shears.",
            "Internal Tools Python Engineer. Experience building CLI utilities, Python package management (Poetry/Pipenv), and automated tooling.",
        ),
        (
            "Hotel Concierge with 5 years experience making dining reservations, booking show tickets, and hailing taxicabs for hotel guests.",
            "Associate Software Engineer. Ideal candidate has solid foundation in Python, algorithms, data structures, and object-oriented programming.",
        ),
        (
            "Security Patrol Guard with 4 years experience walking building hallways, checking locked doors, and writing overnight shift incident logs.",
            "Web Data Extraction Engineer needed. Must have expertise in Python, Scrapy, parsing HTML/JSON, and database integration.",
        ),
        (
            "Car Wash Attendant with 2 years experience hand drying luxury vehicles, cleaning windshields, and vacuuming interior carpets.",
            "Full Stack Python Engineer. Requires Python (Flask or FastAPI), React basics, SQL databases, and web development experience.",
        ),
        (
            "Forklift Operator with 5 years experience driving sit-down forklifts, stacking pallets 4-high, and loading shipping containers.",
            "Embedded Python Developer needed for IoT gateway development. Experience with Python on Linux SBCs, serial communication, and hardware.",
        ),
        (
            "Flight Attendant with 6 years experience serving in-flight meals, checking passenger seatbelts, and handling medical emergencies.",
            "Serverless Python Developer. Strong background in AWS Lambda, Python 3, API Gateway, and event-driven architectures.",
        ),
        (
            "Tailor and Seamstress with 8 years experience hemming trousers, altering formal dresses, and replacing broken zippers.",
            "Software Development Engineer in Test (SDET - Python). Must have advanced Pytest experience, API testing skills, and CI/CD.",
        ),
        (
            "Roofing Repairer with 6 years experience tarring flat commercial roofs, sealing skylight flashings, and replacing damaged shingles.",
            "GIS Python Developer needed to build automated geospatial workflows. Requires Python, GDAL/OGR, ArcPy, and spatial databases.",
        ),
        (
            "Pet Shop Retail Clerk with 3 years experience stocking aquarium fish food, cleaning rodent cages, and answering customer questions.",
            "Distributed Systems Python Engineer. Deep knowledge of message queues, distributed task execution, Redis, and Celery required.",
        ),
        (
            "Janitor with 5 years experience cleaning high school restrooms, mopping hallways, and operating floor wax stripping machines.",
            "Security Automation Engineer. Strong Python scripting, familiarity with security scanning tools, and API integration.",
        ),
        (
            "Massage Therapist with 4 years experience providing Swedish massage and hot stone therapy treatments in a wellness spa.",
            "Game Backend Engineer (Python). Requires asyncio, Redis, WebSocket networking, and relational database management.",
        ),
        (
            "Lawn Mower Repair Mechanic with 5 years experience overhauling small two-stroke engines, sharpening mower blades, and rebuilding carburetors.",
            "Desktop Application Developer wanted. Must be proficient in Python, PyQt or PySide, and multi-threaded GUI design.",
        ),
    ]

    for r, j in python_positives:
        examples.append(DatasetExample(r, j, 1, "Python & Software Development"))
    for r, j in python_negatives:
        examples.append(DatasetExample(r, j, 0, "Python & Software Development"))

    # -------------------------------------------------------------------------
    # DOMAIN 5: Cybersecurity (45 examples)
    # -------------------------------------------------------------------------
    sec_positives = [
        (
            "Cybersecurity Analyst with 3 years experience working in a 24/7 Security Operations Center (SOC). Proficient in monitoring SIEM tools (Splunk, Microsoft Sentinel), analyzing alerts, investigating phishing incidents, and conducting network packet analysis with Wireshark. Familiar with MITRE ATT&CK framework and incident triage procedures.",
            "Seeking a SOC Analyst to monitor and triage security alerts. Requirements: Hands-on experience with SIEM platforms (Splunk or Sentinel), network protocols, Wireshark, triage of malware/phishing alerts, and understanding of the MITRE ATT&CK framework.",
        ),
        (
            "Penetration Tester with 4 years experience performing ethical hacking, web application security assessments, and network vulnerability scans. Proficient in Kali Linux, Burp Suite Professional, Metasploit, Nmap, and OWASP Top 10 vulnerabilities. Certified Ethical Hacker (CEH) and OSCP certified.",
            "Looking for an Application & Network Penetration Tester. Must have proven experience identifying security flaws in web apps and networks using tools like Burp Suite, Nmap, and Metasploit. OSCP certification preferred.",
        ),
        (
            "Information Security Specialist with 5 years experience developing corporate security policies, managing ISO 27001 audits, and conducting vendor risk assessments. Strong knowledge of NIST CSF, GDPR compliance, and security awareness training.",
            "Hiring an Information Security & Compliance Specialist. Qualifications: Deep knowledge of NIST frameworks, ISO 27001, risk management methodologies, and security policy enforcement.",
        ),
        (
            "Junior Cybersecurity Analyst with B.S. in Cybersecurity and CompTIA Security+ certification. Hands-on laboratory experience with Linux security hardening, firewall rule configuration (iptables), and vulnerability scanning using Nessus.",
            "Entry-Level Cybersecurity Associate needed. Requirements: Security+ certification or relevant degree, understanding of TCP/IP networking, fundamental security principles, and familiarity with vulnerability assessment tools.",
        ),
        (
            "Cloud Security Engineer with 4 years experience securing AWS and Azure environments. Implemented IAM least-privilege policies, configured AWS GuardDuty, Security Hub, and automated infrastructure compliance scans using Terraform and checkov.",
            "Cloud Security Engineer wanted. Must have strong expertise in AWS or Azure security controls, IAM architecture, cloud compliance automation, and container security.",
        ),
        (
            "Incident Response and Forensics Investigator with 5 years experience investigating ransomware breaches and APT intrusions. Skilled in memory forensics using Volatility, disk imaging with FTK Imager, and malware analysis.",
            "Digital Forensics & Incident Response (DFIR) Specialist. Required: Hands-on experience investigating security incidents, memory/disk forensics, timeline reconstruction, and malware triage.",
        ),
        (
            "Application Security (AppSec) Engineer with 4 years experience integrating SAST and DAST scanning into CI/CD pipelines using SonarQube, Snyk, and OWASP ZAP. Guided engineering teams on remediating software vulnerabilities.",
            "AppSec Engineer needed. Requirements: Strong understanding of secure coding practices, OWASP Top 10, automated SAST/DAST tooling, and developer vulnerability remediation guidance.",
        ),
        (
            "Vulnerability Management Analyst with 3 years experience coordinating enterprise patching cycles, running Qualys scans, and prioritizing CVEs based on CVSS scores and exploit likelihood.",
            "Vulnerability Management Specialist wanted. Must have experience running Qualys or Tenable Nessus, tracking CVE remediations, and reporting security posture metrics.",
        ),
        (
            "Identity and Access Management (IAM) Engineer with 4 years experience configuring Okta Single Sign-On (SSO), SAML 2.0, OAuth 2.0, and multi-factor authentication (MFA) across enterprise SaaS apps.",
            "IAM Engineer needed. Expertise in enterprise identity providers (Okta / Ping / Azure AD), SAML, OIDC, SCIM provisioning, and privileged access management required.",
        ),
        (
            "Network Security Engineer with 5 years experience deploying Palo Alto Networks next-generation firewalls, configuring IPsec VPN tunnels, and managing intrusion prevention systems (IPS).",
            "Network Security Engineer wanted. Requirements: Extensive experience with Next-Gen Firewalls (Palo Alto, Fortinet), VPN architectures, and network segmentation.",
        ),
        (
            "Security Automation Engineer with 3 years experience building SOAR playbooks in Cortex XSOAR and writing Python automation scripts to enrich threat intelligence indicators (IoCs).",
            "SOAR Automation Developer. Strong Python scripting, REST API integration, and experience designing automated incident response workflows required.",
        ),
        (
            "Cyber Threat Intelligence (CTI) Analyst with 3 years experience tracking threat actor groups, analyzing dark web feeds, and producing actionable intelligence briefs using MISP and OpenCTI.",
            "Threat Intelligence Analyst needed. Experience tracking cyber adversaries, analyzing indicators of compromise (IoCs), and writing threat assessments required.",
        ),
        (
            "Endpoint Security Specialist with 3 years experience deploying and administering CrowdStrike Falcon EDR agents, writing custom detection queries, and remediating endpoint malware infections.",
            "EDR Specialist wanted to manage endpoint protection. Experience administering CrowdStrike, SentinelOne, or Microsoft Defender for Endpoint required.",
        ),
        (
            "Cryptographic Security Engineer experienced in public key infrastructure (PKI), TLS certificate lifecycle management, and hardware security modules (HSM). Strong understanding of asymmetric encryption.",
            "PKI & Cryptography Engineer needed. Deep understanding of x509 certificates, TLS protocols, digital signatures, and encryption algorithms required.",
        ),
        (
            "Senior Governance, Risk, and Compliance (GRC) Analyst with 6 years experience facilitating SOC 2 Type II examinations and managing enterprise risk registers.",
            "GRC Manager needed to lead compliance audits. Requirements: 5+ years experience in SOC 2, HIPAA, or PCI-DSS frameworks, and risk assessment leadership.",
        ),
        (
            "Cybersecurity Intern with coursework in network defense, ethical hacking labs, and Python scripting for port scanners. CompTIA Network+ certified.",
            "Cybersecurity Internship position. Ideal candidate is pursuing a degree in Information Security/CS and understands basic networking, Linux, and security principles.",
        ),
        (
            "Zero Trust Architecture Consultant with 5 years experience implementing micro-segmentation, software-defined perimeters, and conditional access policies.",
            "Zero Trust Security Architect. Experience designing zero-trust frameworks, identity-based access, and network micro-segmentation required.",
        ),
        (
            "DevSecOps Engineer with 3 years experience embedding container security scanning (Trivy, Grype) and policy-as-code (OPA Gatekeeper) into Kubernetes CI/CD workflows.",
            "DevSecOps Engineer to secure continuous delivery pipelines. Must know Docker, Kubernetes, CI/CD tools, and automated vulnerability gate enforcement.",
        ),
        (
            "Malware Reverse Engineer with 4 years experience dissecting malicious binaries using IDA Pro, Ghidra, x64dbg, and analyzing assembly language instructions.",
            "Reverse Engineer / Malware Analyst. Requires proficiency in x86/x64 assembly, Ghidra or IDA Pro, dynamic sandboxing, and unpacking obfuscated executables.",
        ),
        (
            "SCADA & Industrial Control Systems (ICS) Security Specialist with experience auditing operational technology (OT) networks using IEC 62443 standards.",
            "ICS / OT Security Analyst. Experience securing industrial control networks, Modbus/DNP3 protocols, and critical infrastructure environments required.",
        ),
        (
            "Email Security Specialist with 3 years experience configuring SPF, DKIM, DMARC records, and managing Proofpoint / Mimecast anti-phishing gateways.",
            "Email Security Analyst needed. Strong expertise in DMARC/DKIM enforcement, email gateway filtering, and business email compromise (BEC) defense required.",
        ),
        (
            "Physical Security and Access Control Technician with experience installing biometric readers, card badge readers, and security turnstiles.",
            "Physical & Information Security Coordinator. Experience integrating physical badge access with identity databases and auditing facility access logs.",
        ),
        (
            "Cybersecurity Content and Awareness Specialist with 3 years experience creating engaging phishing simulation campaigns and employee training modules using KnowBe4.",
            "Security Awareness Program Coordinator needed to run employee phishing drills and educational security seminars.",
        ),
    ]

    sec_negatives = [
        (
            "Landscape Gardener with 5 years experience laying turf sod, trimming decorative trees, and installing backyard sprinkler pipes.",
            "Seeking a SOC Analyst to monitor and triage security alerts. Requirements: Hands-on experience with SIEM platforms (Splunk), network protocols, and Wireshark.",
        ),
        (
            "Drywaller with 6 years experience taping sheetrock joints, applying plaster mud, and sanding smooth finishes in office suites.",
            "Looking for an Application & Network Penetration Tester. Must have proven experience identifying security flaws using Burp Suite, Nmap, and Metasploit.",
        ),
        (
            "Bakery Counter Server with 3 years experience slicing artisan bread loaves, bagging rolls, and managing the cash register.",
            "Hiring an Information Security & Compliance Specialist. Qualifications: Deep knowledge of NIST frameworks, ISO 27001, and risk management.",
        ),
        (
            "Automotive Detailer with 4 years experience shampooing car carpets, polishing exterior chrome trim, and applying protective wax.",
            "Entry-Level Cybersecurity Associate needed. Requirements: Security+ certification or relevant degree, understanding of TCP/IP networking, and security basics.",
        ),
        (
            "House Painter with 7 years experience scraping old paint, masking window borders, and rolling interior walls with flat latex.",
            "Cloud Security Engineer wanted. Must have strong expertise in AWS or Azure security controls, IAM architecture, and cloud compliance.",
        ),
        (
            "Bartender with 5 years experience mixing cocktails, tapping beer kegs, and wiping down wooden bar counters during happy hour.",
            "Digital Forensics & Incident Response (DFIR) Specialist. Required: Hands-on experience investigating security incidents, memory/disk forensics, and malware triage.",
        ),
        (
            "Plumbing Laborer with 3 years experience carrying cast iron pipes, digging sewer trenches, and cleaning plumbing snake cables.",
            "AppSec Engineer needed. Requirements: Strong understanding of secure coding practices, OWASP Top 10, automated SAST/DAST tooling.",
        ),
        (
            "Hair Stylist with 6 years experience cutting layered bob hairstyles, applying hair color foils, and styling bridal up-dos.",
            "Vulnerability Management Specialist wanted. Must have experience running Qualys or Tenable Nessus, and tracking CVE remediations.",
        ),
        (
            "Hotel Maid with 4 years experience changing bed sheets, replenishing bathroom towels, and vacuuming guest room carpets.",
            "IAM Engineer needed. Expertise in enterprise identity providers (Okta / Ping / Azure AD), SAML, OIDC, and privileged access management.",
        ),
        (
            "Bus Driver with 8 years experience driving city buses, announcing scheduled passenger stops, and assisting disabled riders.",
            "Network Security Engineer wanted. Requirements: Extensive experience with Next-Gen Firewalls (Palo Alto), and VPN architectures.",
        ),
        (
            "Shoe Store Associate with 2 years experience measuring customer shoe sizes, checking backroom inventory, and ringing up sales.",
            "SOAR Automation Developer. Strong Python scripting, REST API integration, and automated incident response workflows required.",
        ),
        (
            "Carpet Cleaner with 5 years experience operating truck-mounted steam extraction machines and pre-treating stubborn carpet stains.",
            "Threat Intelligence Analyst needed. Experience tracking cyber adversaries, analyzing indicators of compromise (IoCs), and writing threat assessments.",
        ),
        (
            "Dental Hygienist with 4 years experience scaling dental tartar, applying fluoride gels, and polishing adult teeth in a dental clinic.",
            "EDR Specialist wanted to manage endpoint protection. Experience administering CrowdStrike, SentinelOne, or Microsoft Defender.",
        ),
        (
            "Forklift Driver with 4 years experience moving palletized goods, operating shrink-wrap machines, and loading boxcars.",
            "PKI & Cryptography Engineer needed. Deep understanding of x509 certificates, TLS protocols, digital signatures, and encryption.",
        ),
        (
            "Pet Sitter and Dog Walker with 3 years experience feeding domestic cats, walking dogs twice daily, and administering pet oral medication.",
            "GRC Manager needed to lead compliance audits. Requirements: 5+ years experience in SOC 2, HIPAA, or PCI-DSS frameworks.",
        ),
        (
            "Flight Attendant with 5 years experience performing safety demonstrations, serving coffee, and assisting passengers during turbulence.",
            "Cybersecurity Internship position. Ideal candidate is pursuing a degree in Information Security/CS and understands basic networking.",
        ),
        (
            "Upholstery Cleaner with 4 years experience steam cleaning microfiber couches, removing wine stains, and deodorizing armchairs.",
            "Zero Trust Security Architect. Experience designing zero-trust frameworks, identity-based access, and network micro-segmentation.",
        ),
        (
            "Automotive Tire Changer with 3 years experience dismounting flat tires, plugging punctures, and torquing wheel lug nuts.",
            "DevSecOps Engineer to secure continuous delivery pipelines. Must know Docker, Kubernetes, CI/CD tools, and automated vulnerability gates.",
        ),
        (
            "Meat Slicer in a Delicatessen with 4 years experience operating electric deli slicers, weighing cold cuts, and wrapping meats in butcher paper.",
            "Reverse Engineer / Malware Analyst. Requires proficiency in x86/x64 assembly, Ghidra or IDA Pro, and dynamic sandboxing.",
        ),
        (
            "Commercial Window Washer with 6 years experience squeegeeing exterior glass panes from hanging high-rise scaffold stages.",
            "ICS / OT Security Analyst. Experience securing industrial control networks, Modbus/DNP3 protocols, and critical infrastructure environments.",
        ),
        (
            "Golf Caddy with 3 years experience carrying golf bags, raking sand traps, and providing distance yardages on country club greens.",
            "Email Security Analyst needed. Strong expertise in DMARC/DKIM enforcement, email gateway filtering, and anti-phishing defense.",
        ),
        (
            "Swimming Pool Attendant with 2 years experience skimming surface leaves, handing out sun loungers, and monitoring pool rules.",
            "Security Awareness Program Coordinator needed to run employee phishing drills and educational security seminars.",
        ),
    ]

    for r, j in sec_positives:
        examples.append(DatasetExample(r, j, 1, "Cybersecurity"))
    for r, j in sec_negatives:
        examples.append(DatasetExample(r, j, 0, "Cybersecurity"))

    # -------------------------------------------------------------------------
    # DOMAIN 6: Cloud Computing and DevOps (45 examples)
    # -------------------------------------------------------------------------
    cloud_positives = [
        (
            "DevOps Engineer with 4 years experience implementing infrastructure as code (IaC) with Terraform, managing Kubernetes clusters on AWS (EKS), and building automated CI/CD pipelines using GitLab CI and GitHub Actions. Proficient in Docker, Linux shell scripting, and Prometheus/Grafana monitoring.",
            "Hiring a DevOps / Cloud Platform Engineer. Requirements: Strong experience with Terraform, Kubernetes (K8s), Docker containerization, AWS cloud services (EKS, EC2, S3), CI/CD automation, and Linux administration.",
        ),
        (
            "Cloud Solutions Architect with 7 years experience designing multi-region, fault-tolerant infrastructure in AWS. Deep expertise in VPC peering, Direct Connect, IAM security, Auto Scaling, RDS PostgreSQL, and cost optimization (FinOps). AWS Certified Solutions Architect Professional.",
            "Senior Cloud Architect wanted to guide enterprise cloud migration. Must have 6+ years in AWS architectural design, hybrid cloud networking, infrastructure security, and cost governance.",
        ),
        (
            "Site Reliability Engineer (SRE) with 4 years experience managing high-availability production systems, setting Service Level Objectives (SLOs), and automating incident mitigation using Python and Bash. Skilled in Datadog, PagerDuty, and distributed tracing.",
            "Looking for a Site Reliability Engineer (SRE). Must have strong skills in Linux systems internals, observability (Prometheus, Grafana, Datadog), incident management, and automation in Python or Go.",
        ),
        (
            "Junior DevOps Engineer with B.Tech in IT. Hands-on experience setting up Docker containers, basic GitHub Actions CI/CD workflows, and hosting static web applications on AWS S3 and CloudFront. Familiar with Linux commands and Git.",
            "Junior Cloud & DevOps Associate to support deployment operations. Requirements: Understanding of Linux, Docker basics, Git, basic AWS services, and passion for continuous integration.",
        ),
        (
            "Kubernetes Platform Engineer with 4 years experience administering multi-tenant bare-metal and cloud Kubernetes clusters. Experienced with Helm package management, ArgoCD GitOps, and Istio service mesh configuration.",
            "Platform Engineer needed with deep Kubernetes expertise. Must know Helm, GitOps workflows (ArgoCD or Flux), service mesh (Istio), and cluster security.",
        ),
        (
            "Azure Cloud Engineer with 4 years experience managing Azure subscriptions, Azure DevOps pipelines, virtual networks (VNet), and Azure Kubernetes Service (AKS). Skilled in ARM templates and Bicep.",
            "Azure Infrastructure Engineer wanted. Requirements: Experience with Azure cloud services, Azure DevOps CI/CD, AKS, Bicep/Terraform, and Windows/Linux server administration.",
        ),
        (
            "Build & Release Engineer with 3 years experience maintaining Maven, Gradle, and npm automated build pipelines in Jenkins. Reduced pipeline build durations by 35% through artifact caching and parallel execution.",
            "Build & Release Engineer needed to manage enterprise compilation and packaging pipelines. Strong Jenkins, Groovy pipeline scripting, and build tool familiarity required.",
        ),
        (
            "Google Cloud Platform (GCP) Specialist with 4 years experience configuring Google Kubernetes Engine (GKE), Cloud IAM, BigQuery networking, and VPC Service Controls. GCP Professional Cloud Architect certified.",
            "GCP Cloud Engineer needed. Experience managing Google Cloud Platform environments, GKE, Cloud Functions, and automated Terraform deployments required.",
        ),
        (
            "Infrastructure Automation Engineer with 3 years experience automating server configuration using Ansible playbooks and managing baseline VM images with HashiCorp Packer.",
            "Automation Engineer wanted. Must be skilled in Ansible configuration management, Packer, Linux system administration, and infrastructure provisioning.",
        ),
        (
            "Cloud Security & Compliance DevOps Engineer experienced in enforcing compliance as code with Checkov, tfsec, and AWS Config. Automated security gate checks in GitLab CI.",
            "DevSecOps / Cloud Compliance Specialist needed to embed automated security testing into Terraform and Docker build pipelines.",
        ),
        (
            "Monitoring & Observability Specialist with 3 years experience deploying Prometheus, Alertmanager, Grafana, and OpenTelemetry collectors across Kubernetes microservices.",
            "Observability Engineer wanted. Deep experience building Grafana dashboards, tuning Prometheus alerting thresholds, and configuring distributed tracing required.",
        ),
        (
            "Entry-Level Cloud Support Associate with AWS Certified Cloud Practitioner. Handled basic cloud ticket resolutions, user access provisioning, and EBS volume snapshots.",
            "Cloud Support Technician. Ideal for entry-level candidates with basic cloud certifications, good troubleshooting mindset, and familiarity with cloud console operations.",
        ),
        (
            "Storage & Backup Cloud Engineer with 4 years experience managing enterprise S3 storage tiering, AWS Glacier archives, and disaster recovery replication runbooks.",
            "Cloud Storage & Disaster Recovery Engineer. Requires strong knowledge of cloud storage protocols, lifecycle policies, and disaster recovery testing.",
        ),
        (
            "Serverless Cloud Developer with 3 years experience architecting event-driven microservices using AWS Lambda, EventBridge, DynamoDB, and Serverless Framework.",
            "Serverless Platform Engineer needed. Strong background in AWS Lambda, event-driven design patterns, API Gateway, and automated CI/CD.",
        ),
        (
            "Linux Systems Administrator transitioning to DevOps with 6 years experience in Red Hat Enterprise Linux (RHEL), Bash scripting, systemd services, and Apache/Nginx web servers.",
            "Linux / DevOps Engineer needed. Strong Linux fundamentals, shell scripting, troubleshooting, and eagerness to expand into Docker and Terraform required.",
        ),
        (
            "Database DevOps Engineer experienced in automating PostgreSQL and MySQL database schema migrations in CI/CD using Flyway and Liquibase with zero-downtime Blue/Green deployments.",
            "Database Reliability Engineer (DBRE). Experience automating database migrations, managing replication failovers, and cloud database optimization required.",
        ),
        (
            "CI/CD Pipeline Specialist with 3 years experience writing reusable GitHub Actions composite workflows and optimizing Docker multi-stage builds for container registry storage.",
            "CI/CD Engineer wanted. Must have advanced knowledge of GitHub Actions, Docker optimization, secrets management, and deployment pipelines.",
        ),
        (
            "Cloud Networking Specialist with 5 years experience designing BGP routing, transit gateways, AWS DirectConnect, and SD-WAN integrations.",
            "Cloud Network Engineer needed. Deep understanding of routing protocols, IP addressing, cloud VPC architectures, and firewall appliances required.",
        ),
        (
            "FinOps / Cloud Cost Optimization Analyst with 3 years experience analyzing AWS Cost Explorer reports, managing Reserved Instances, and optimizing cloud compute spend.",
            "Cloud FinOps Analyst. Experience tracking cloud spending metrics, rightsizing workloads, and implementing cost optimization strategies required.",
        ),
        (
            "Edge Cloud DevOps Engineer experienced in deploying containerized microservices to remote IoT gateways using K3s, Docker, and Ansible.",
            "Edge DevOps Engineer. Experience managing lightweight Kubernetes (K3s), edge device deployment, and remote connectivity required.",
        ),
        (
            "High-Performance Computing (HPC) Cloud Engineer with experience configuring GPU compute clusters on AWS with Slurm and Elastic Fabric Adapter (EFA).",
            "HPC Cloud Engineer needed. Experience configuring GPU instances, high-throughput parallel filesystems (Lustre), and Slurm on AWS or GCP required.",
        ),
        (
            "Senior DevOps Manager with 8 years experience leading global infrastructure teams, driving cloud adoption, and maintaining 99.99% system availability.",
            "Director / Manager of Cloud Engineering to lead a team of 10+ SRE and DevOps engineers across multi-cloud environments.",
        ),
        (
            "Junior Linux & Cloud Intern with experience spinning up Ubuntu EC2 instances, setting up SSH key pairs, and running Python scripts via crontab.",
            "DevOps Internship. Opportunity for students with practical Linux familiarity and basic cloud knowledge to learn production infrastructure.",
        ),
    ]

    cloud_negatives = [
        (
            "House Painter with 5 years experience masking trim, filling nail holes with spackle, and rolling ceilings with eggshell paint.",
            "Hiring a DevOps / Cloud Platform Engineer. Requirements: Strong experience with Terraform, Kubernetes, Docker, and AWS.",
        ),
        (
            "Drywaller with 4 years experience lifting sheetrock, cutting electrical box openings, and taping joints with mesh tape.",
            "Senior Cloud Architect wanted. Must have 6+ years in AWS architectural design, hybrid cloud networking, and cost governance.",
        ),
        (
            "Hairdresser with 6 years experience performing shampoo blowdrys, applying foil highlights, and trimming split ends.",
            "Looking for a Site Reliability Engineer (SRE). Must have strong skills in Linux systems internals, observability (Prometheus), and Python.",
        ),
        (
            "Bartender with 3 years experience pouring draft beers, polishing glassware, and restocking beer coolers during evening shifts.",
            "Junior Cloud & DevOps Associate to support deployment operations. Requirements: Linux, Docker basics, Git, and basic AWS services.",
        ),
        (
            "Bakery Assistant with 4 years experience packaging dinner rolls, kneading pie dough, and wiping down stainless steel mixing bowls.",
            "Platform Engineer needed with deep Kubernetes expertise. Must know Helm, GitOps workflows (ArgoCD), and service mesh (Istio).",
        ),
        (
            "Automotive Detailer with 5 years experience buffing car clearcoats, cleaning leather seats, and spraying tire shine.",
            "Azure Infrastructure Engineer wanted. Requirements: Experience with Azure cloud services, Azure DevOps CI/CD, and AKS.",
        ),
        (
            "Plumbing Laborer with 4 years experience carrying cast iron pipes, replacing faucet washers, and snaking shower drains.",
            "Build & Release Engineer needed to manage enterprise compilation pipelines. Strong Jenkins and Groovy scripting required.",
        ),
        (
            "Hotel Maid with 3 years experience vacuuming hotel room carpets, making king beds, and sanitizing bathroom fixtures.",
            "GCP Cloud Engineer needed. Experience managing Google Cloud Platform environments, GKE, and automated Terraform deployments.",
        ),
        (
            "Bus Driver with 6 years experience operating commuter buses, monitoring passenger fares, and adhering to strict route timetables.",
            "Automation Engineer wanted. Must be skilled in Ansible configuration management, Packer, and Linux administration.",
        ),
        (
            "Forklift Operator with 5 years experience driving pallet jacks, loading flatbed trailers, and stacking wooden crates in warehouses.",
            "DevSecOps / Cloud Compliance Specialist needed to embed automated security testing into Terraform and Docker build pipelines.",
        ),
        (
            "Landscape Laborer with 4 years experience mowing soccer fields, weedwhacking fence lines, and raking autumn leaves.",
            "Observability Engineer wanted. Deep experience building Grafana dashboards and tuning Prometheus alerting thresholds required.",
        ),
        (
            "Pet Groomer with 3 years experience bathing golden retrievers, trimming dog nails, and brushing thick animal coats.",
            "Cloud Storage & Disaster Recovery Engineer. Requires strong knowledge of cloud storage protocols and disaster recovery testing.",
        ),
        (
            "Shoe Salesperson with 2 years experience retrieving sneaker boxes, organizing retail displays, and running register sales.",
            "Serverless Platform Engineer needed. Strong background in AWS Lambda, event-driven design patterns, and DynamoDB.",
        ),
        (
            "Carpet Installer with 7 years experience cutting carpet padding, seaming carpet joints, and tucking edges into baseboard tack strips.",
            "Linux / DevOps Engineer needed. Strong Linux fundamentals, shell scripting, troubleshooting, and Docker.",
        ),
        (
            "Flight Attendant with 5 years experience delivering pre-flight safety briefings, serving tea and coffee, and calming nervous passengers.",
            "Database Reliability Engineer (DBRE). Experience automating database migrations, managing replication failovers, and cloud databases.",
        ),
        (
            "Upholsterer with 6 years experience stapling fabric over chair frames, tying coil springs, and sewing decorative welt cord.",
            "CI/CD Engineer wanted. Must have advanced knowledge of GitHub Actions, Docker optimization, and deployment pipelines.",
        ),
        (
            "Tire Technician with 4 years experience patching tire tread punctures, balancing steel rims, and checking wheel alignment.",
            "Cloud Network Engineer needed. Deep understanding of routing protocols, IP addressing, and cloud VPC architectures.",
        ),
        (
            "Commercial Window Cleaner with 5 years experience washing second-story office glass using water-fed extension poles.",
            "Cloud FinOps Analyst. Experience tracking cloud spending metrics, rightsizing workloads, and cost optimization strategies.",
        ),
        (
            "Deli Counter Worker with 3 years experience slicing provolone cheese, wrapping cold cut sandwiches, and stocking beverage cases.",
            "Edge DevOps Engineer. Experience managing lightweight Kubernetes (K3s), edge device deployment, and remote connectivity.",
        ),
        (
            "Swimming Pool Attendant with 2 years experience testing pool water alkalinity, vacuuming pool bottom sediment, and rolling pool covers.",
            "HPC Cloud Engineer needed. Experience configuring GPU instances, high-throughput parallel filesystems, and Slurm.",
        ),
        (
            "Golf Caddy with 4 years experience carrying two golf bags per round, replacing divots, and cleaning golf irons with wet towels.",
            "Director / Manager of Cloud Engineering to lead a team of 10+ SRE and DevOps engineers across multi-cloud environments.",
        ),
        (
            "Piano Mover with 6 years experience strapping baby grand pianos onto moving skids and hoisting them through window balconies.",
            "DevOps Internship. Opportunity for students with practical Linux familiarity and basic cloud knowledge to learn production infrastructure.",
        ),
    ]

    for r, j in cloud_positives:
        examples.append(DatasetExample(r, j, 1, "Cloud & DevOps"))
    for r, j in cloud_negatives:
        examples.append(DatasetExample(r, j, 0, "Cloud & DevOps"))

    # -------------------------------------------------------------------------
    # DOMAIN 7: Database and Business Intelligence (45 examples)
    # -------------------------------------------------------------------------
    db_positives = [
        (
            "Database Administrator (DBA) with 5 years experience administering mission-critical PostgreSQL and MySQL production clusters. Proficient in query execution plan optimization (EXPLAIN ANALYZE), index tuning (B-Tree, GIN), automated pg_dump backups, and streaming replication setup. Strong knowledge of Linux server performance monitoring.",
            "Seeking an experienced PostgreSQL / MySQL Database Administrator to maintain high-availability database clusters. Requirements: Deep knowledge of SQL query optimization, database backup/recovery, replication configurations, and performance tuning.",
        ),
        (
            "Senior Data Warehouse Engineer with 6 years experience designing dimensional schemas (Star Schema, Snowflake Schema) in Snowflake and Amazon Redshift. Expert in building data transformation pipelines using dbt, SQL, and Apache Airflow.",
            "Hiring a Data Warehouse Specialist. Must have strong expertise in Snowflake or BigQuery, dimensional data modeling, building dbt models, and optimizing complex analytical SQL queries.",
        ),
        (
            "Business Intelligence Developer with 4 years experience creating enterprise Power BI and Tableau dashboards. Skilled in writing advanced DAX measures, Power Query M code, and integrating diverse ERP and CRM data sources.",
            "BI Developer needed to deliver executive reporting suites. Key qualifications: Advanced Power BI experience, DAX formulas, data modeling, SQL query writing, and dashboard design.",
        ),
        (
            "Junior SQL Developer with 1 year experience writing complex SELECT queries, inner/outer joins, aggregations, and subqueries in SQL Server. Built automated monthly financial summary reports in Excel and SSRS.",
            "Looking for a Junior SQL / Reporting Analyst. Requirements: Strong SQL fundamentals (joins, group by, unions), basic understanding of relational database design, and Excel proficiency.",
        ),
        (
            "ETL Pipeline Developer with 4 years experience building automated batch data ingestion workflows using Talend, Python, and SQL Server Integration Services (SSIS). Managed data validation checks and error logging tables.",
            "ETL Developer wanted to build data migration pipelines. Must have experience with SSIS or Talend, advanced T-SQL stored procedures, and data cleansing practices.",
        ),
        (
            "Oracle Database Administrator with 6 years experience supporting Oracle 19c enterprise databases, Oracle RAC clusters, and Data Guard standby configurations. Proficient in PL/SQL programming and RMAN backup scripts.",
            "Oracle DBA position. Requires 5+ years experience with Oracle Database administration, PL/SQL, RMAN backup strategies, and high-availability architecture.",
        ),
        (
            "Data Modeler with 5 years experience designing enterprise logical and physical data models using Erwin Data Modeler and ER/Studio. Strong understanding of 3NF relational normalization and Data Vault 2.0 methodology.",
            "Enterprise Data Modeler needed. Strong expertise in relational normalization, conceptual-to-physical data modeling, and Data Vault architectures required.",
        ),
        (
            "NoSQL Database Specialist with 3 years experience deploying and scaling MongoDB and Cassandra clusters. Modeled document collections, tuned sharding keys, and optimized read/write throughput for high-volume logs.",
            "NoSQL Database Engineer wanted. Experience administering MongoDB, document modeling, sharding, and replica set maintenance required.",
        ),
        (
            "Looker / Business Intelligence Specialist with 3 years experience creating reusable LookML models, explores, and interactive dashboards. Partnered with product and sales operations teams to define key performance indicators.",
            "Looker BI Analyst needed to scale business reporting. Strong LookML development experience, SQL mastery, and metric visualization skills required.",
        ),
        (
            "SQL Server Database Administrator (MS SQL DBA) with 5 years experience configuring AlwaysOn Availability Groups, SQL Agent jobs, and tuning T-SQL index fragmentation.",
            "SQL Server DBA wanted. Requirements: Deep knowledge of Microsoft SQL Server, AlwaysOn clustering, transaction log shipping, and performance troubleshooting.",
        ),
        (
            "Data Governance and Metadata Analyst with 3 years experience cataloging enterprise data assets, documenting data dictionaries in Collibra, and defining data lineage workflows.",
            "Data Governance Analyst needed to implement data quality frameworks. Experience with data catalog tools, metadata management, and data profiling required.",
        ),
        (
            "Entry-Level Data Reporting Assistant with B.S. in Information Systems. Completed coursework in relational databases, SQL queries, and building student record reports in MySQL.",
            "Entry-Level Reporting Specialist. Ideal candidate knows relational database fundamentals, SQL SELECT statements, and enjoys turning numbers into clean charts.",
        ),
        (
            "Cloud Data Migration Engineer with 4 years experience migrating on-premises Oracle and SQL Server databases to AWS Aurora PostgreSQL using AWS Database Migration Service (DMS).",
            "Database Migration Engineer needed. Experience moving relational databases to cloud environments (Aurora/RDS) and using migration tools required.",
        ),
        (
            "Qlik Sense / BI Developer with 4 years experience designing interactive analytics apps, writing Qlik script for data transformation, and managing Qlik Management Console permissions.",
            "Qlik Sense Developer wanted. Must have experience building associative Qlik models, complex scripting, and enterprise dashboard delivery.",
        ),
        (
            "Graph Database Developer with 3 years experience modeling interconnected data networks in Neo4j, writing optimized Cypher queries, and integrating with Python backends.",
            "Graph Database Engineer needed. Experience with Neo4j, Cypher query language, graph data modeling, and performance tuning required.",
        ),
        (
            "Data Quality Analyst with 3 years experience running automated SQL data profiling scripts to identify null anomalies, duplicate customer records, and format inconsistencies.",
            "Data Quality Specialist. Must have strong SQL data validation skills, experience building automated reconciliation checks, and reporting error rates.",
        ),
        (
            "Redis / In-Memory Database Administrator with experience managing high-throughput Redis clusters, Sentinel failover mechanisms, and memory eviction policies.",
            "In-Memory Database Specialist. Knowledge of Redis data structures, caching patterns, replication, and memory tuning required.",
        ),
        (
            "Tableau Server Administrator with 3 years experience managing user permissions, published data source extracts, and schedule refreshes on Tableau Server / Cloud.",
            "Tableau Administrator needed to manage enterprise visual reporting infrastructure. Experience managing Tableau sites, governance, and permissions required.",
        ),
        (
            "Time-Series Database Engineer with 3 years experience managing TimescaleDB and InfluxDB instances for industrial telemetry data storage and hypertable partition optimization.",
            "Time-Series Database Specialist. Experience with TimescaleDB, InfluxDB, metric aggregation queries, and retention policies required.",
        ),
        (
            "Master Data Management (MDM) Specialist with 4 years experience implementing customer golden record consolidation rules and resolving identity deduplication.",
            "MDM Specialist wanted. Experience in master data management architectures, entity deduplication matching rules, and data stewardship required.",
        ),
        (
            "Spatial Database Administrator with 3 years experience supporting PostGIS extensions on PostgreSQL, spatial indexing (GIST), and optimizing geographic boundary intersection queries.",
            "Spatial Database Engineer. Strong PostgreSQL and PostGIS skills, spatial indexing, and geographic query optimization required.",
        ),
        (
            "Junior BI Developer with 1 year experience creating sales performance charts in Power BI and writing basic SQL joins to pull transaction rows from PostgreSQL.",
            "Junior BI Developer to support financial reporting team. Must know basic SQL, Power BI chart design, and possess good attention to numerical detail.",
        ),
        (
            "Mainframe Database Specialist with experience managing IBM Db2 on z/OS systems, performing table reorganization utilities, and tuning legacy COBOL SQL calls.",
            "Db2 Database Administrator for enterprise legacy banking modernization. Experience with IBM Db2, database maintenance, and mainframe environments required.",
        ),
    ]

    db_negatives = [
        (
            "House Painter with 5 years experience scraping exterior window trims, painting siding, and pressure washing decks.",
            "Seeking an experienced PostgreSQL / MySQL Database Administrator to maintain high-availability database clusters. Requirements: SQL tuning and backup.",
        ),
        (
            "Drywaller with 6 years experience cutting drywall boards, taping drywall seams, and coating nail heads with joint compound.",
            "Hiring a Data Warehouse Specialist. Must have strong expertise in Snowflake or BigQuery, dimensional data modeling, and dbt models.",
        ),
        (
            "Hairdresser with 4 years experience washing hair, cutting men's fade cuts, and styling women's bridal hair extensions.",
            "BI Developer needed to deliver executive reporting suites. Key qualifications: Advanced Power BI experience, DAX formulas, and SQL.",
        ),
        (
            "Bartender with 5 years experience cutting lemon garnishes, shaking cocktails, and collecting drink payments at a nightclub.",
            "Looking for a Junior SQL / Reporting Analyst. Requirements: Strong SQL fundamentals, understanding of relational databases, and Excel.",
        ),
        (
            "Bakery Assistant with 3 years experience dusting flour over dough tables, measuring sugar sacks, and washing baking sheets.",
            "ETL Developer wanted to build data migration pipelines. Must have experience with SSIS or Talend, advanced T-SQL stored procedures, and cleansing.",
        ),
        (
            "Automotive Detailer with 4 years experience vacuuming car trunks, wiping plastic dashboard vents, and polishing headlamp covers.",
            "Oracle DBA position. Requires 5+ years experience with Oracle Database administration, PL/SQL, RMAN backup strategies, and RAC.",
        ),
        (
            "Plumbing Laborer with 5 years experience carrying PVC fittings, uncoiling copper tubing, and digging water pipe ditches.",
            "Enterprise Data Modeler needed. Strong expertise in relational normalization, conceptual-to-physical data modeling, and Data Vault.",
        ),
        (
            "Hotel Maid with 3 years experience stripping duvet covers, restocking mini-soaps in bathrooms, and running hallway laundry carts.",
            "NoSQL Database Engineer wanted. Experience administering MongoDB, document modeling, sharding, and replica set maintenance.",
        ),
        (
            "Bus Driver with 7 years experience driving cross-town transit buses, checking passenger passes, and keeping scheduled timing.",
            "Looker BI Analyst needed to scale business reporting. Strong LookML development experience, SQL mastery, and metric visualization.",
        ),
        (
            "Forklift Operator with 4 years experience maneuvering warehouse aisles, stacking pallet loads, and staging freight on loading docks.",
            "SQL Server DBA wanted. Requirements: Deep knowledge of Microsoft SQL Server, AlwaysOn clustering, transaction logs, and tuning.",
        ),
        (
            "Landscape Laborer with 4 years experience edging garden lawns, blowing grass clippings off sidewalks, and planting annual flowers.",
            "Data Governance Analyst needed to implement data quality frameworks. Experience with data catalog tools, metadata management, and profiling.",
        ),
        (
            "Pet Groomer with 3 years experience washing cocker spaniels, clipping poodle fur, and cleaning animal ears.",
            "Cloud Data Migration Engineer needed. Experience moving relational databases to cloud environments (Aurora/RDS) and migration tools.",
        ),
        (
            "Shoe Salesperson with 2 years experience fitting running shoes on customers, organizing shelf stock, and processing cash returns.",
            "Qlik Sense Developer wanted. Must have experience building associative Qlik models, complex scripting, and enterprise dashboards.",
        ),
        (
            "Carpet Installer with 6 years experience laying underlayment foam, rolling carpet bolts, and trimming edges along baseboards.",
            "Graph Database Engineer needed. Experience with Neo4j, Cypher query language, graph data modeling, and performance tuning.",
        ),
        (
            "Flight Attendant with 5 years experience passing out snack boxes, demonstrating oxygen masks, and ensuring tray tables are stowed.",
            "Data Quality Specialist. Must have strong SQL data validation skills, experience building automated reconciliation checks, and error logs.",
        ),
        (
            "Upholsterer with 4 years experience cutting velvet upholstery fabric, padding sofa frames with batting, and hammering decorative tacks.",
            "In-Memory Database Specialist. Knowledge of Redis data structures, caching patterns, replication, and memory tuning.",
        ),
        (
            "Tire Technician with 3 years experience dismounting worn tires on tire changers, patching leaks, and balancing alloy wheels.",
            "Tableau Administrator needed to manage enterprise visual reporting infrastructure. Experience managing Tableau sites and governance.",
        ),
        (
            "Commercial Window Cleaner with 4 years experience cleaning ground-floor shop windows with squeegees and soapy water buckets.",
            "Time-Series Database Specialist. Experience with TimescaleDB, InfluxDB, metric aggregation queries, and retention policies.",
        ),
        (
            "Deli Counter Worker with 3 years experience slicing roast beef to order, wrapping cheese wedges, and cleaning meat display cases.",
            "MDM Specialist wanted. Experience in master data management architectures, entity deduplication matching rules, and stewardship.",
        ),
        (
            "Swimming Pool Attendant with 2 years experience scooping pool bugs with long poles, testing chlorine drops, and stacking pool chairs.",
            "Spatial Database Engineer. Strong PostgreSQL and PostGIS skills, spatial indexing, and geographic query optimization.",
        ),
        (
            "Golf Caddy with 3 years experience tracking golf ball drives, handing drivers to club members, and repairing green pitch marks.",
            "Junior BI Developer to support financial reporting team. Must know basic SQL, Power BI chart design, and detail orientation.",
        ),
        (
            "Piano Mover with 5 years experience strapping pianos to piano dollies, moving them through narrow hallways, and loading moving vans.",
            "Db2 Database Administrator for enterprise legacy banking modernization. Experience with IBM Db2 and mainframe environments.",
        ),
    ]

    for r, j in db_positives:
        examples.append(DatasetExample(r, j, 1, "Database & BI"))
    for r, j in db_negatives:
        examples.append(DatasetExample(r, j, 0, "Database & BI"))

    # -------------------------------------------------------------------------
    # DOMAIN 8: General IT and Entry-Level Technology Roles (45 examples)
    # -------------------------------------------------------------------------
    it_positives = [
        (
            "IT Support Specialist with 3 years experience providing Tier 1 and Tier 2 technical helpdesk support for 400+ onsite and remote users. Proficient in Windows 10/11, macOS, Microsoft 365 administration, Active Directory user provisioning, and hardware troubleshooting. Resolved ticketing queues in ServiceNow with 96% first-contact resolution.",
            "Seeking an IT Helpdesk Support Specialist to deliver technical desktop support. Requirements: Experience with Windows and macOS operating systems, Active Directory user management, Microsoft 365, hardware troubleshooting, and IT ticketing systems (ServiceNow / Jira).",
        ),
        (
            "Junior QA Tester with 2 years experience performing manual functional testing, regression testing, and bug logging for web and mobile applications using Jira and TestRail. Familiar with basic Selenium test scripts and Chrome DevTools.",
            "Hiring a Junior Quality Assurance (QA) Tester. Must have experience writing test scenarios, executing manual regression test passes, reporting detailed bug repro steps in Jira, and basic familiarity with automated testing.",
        ),
        (
            "Systems Support Administrator with 4 years experience administering local area networks (LAN), managing office Wi-Fi access points, configuring VoIP phones, and deploying workstation software via Microsoft Intune.",
            "Systems & Network Support Administrator wanted. Requirements: Experience administering office networks, DHCP/DNS setup, VoIP, and managing endpoint deployments using Microsoft Intune or SCCM.",
        ),
        (
            "IT Helpdesk Technician with CompTIA A+ certification and 1 year experience resolving hardware, printer, and software issues for corporate staff. Handled password resets, VPN connection troubleshooting, and workstation imaging.",
            "Entry-Level IT Technician needed. CompTIA A+ or similar certification preferred. Strong customer service mindset, basic computer hardware knowledge, and problem-solving skills required.",
        ),
        (
            "Recent College Graduate with B.S. in Information Technology. Completed coursework in computer networks, Linux fundamentals, database management, and Python scripting. Built home lab running VirtualBox VMs and Active Directory domain.",
            "IT Graduate Associate wanted for rotational technology program. Requirements: Degree in IT/Computer Science, understanding of networking fundamentals, basic Linux/Windows knowledge, and eagerness to learn.",
        ),
        (
            "Desktop Support Technician with 3 years experience imaging Windows laptops with PXE boot, replacing broken laptop screens, and managing corporate asset inventory in Snipe-IT.",
            "Desktop Hardware Support Specialist needed to manage laptop fleet. Experience with hardware diagnostics, component replacements, operating system imaging, and asset tracking required.",
        ),
        (
            "Junior Automation QA with 1 year experience writing automated end-to-end browser tests using Cypress and JavaScript. Documented API responses with Postman and verified database records.",
            "Junior QA Automation Engineer. Qualifications: Basic JavaScript skills, experience writing UI tests in Cypress or Selenium, and understanding of software QA lifecycles.",
        ),
        (
            "Technical Customer Support Representative with 3 years experience troubleshooting SaaS software issues via chat and email in Zendesk. Assisted non-technical users with browser configuration and account permissions.",
            "Technical Support Representative needed for software company. Must have excellent written communication, patient troubleshooting abilities, and experience resolving software usage tickets.",
        ),
        (
            "Field Service Technician with 3 years experience installing commercial point-of-sale (POS) terminals, barcode scanners, and receipt printers in retail stores. Managed Cat6 cabling and network terminations.",
            "Field IT Deployment Technician wanted. Experience installing retail hardware, running Ethernet cabling, and configuring network peripherals required.",
        ),
        (
            "IT Asset Management Coordinator with 2 years experience tracking laptop lifecycles, processing hardware returns from departing employees, and coordinating equipment shipping logistics.",
            "IT Procurement & Asset Coordinator. Experience managing computer hardware inventory, working with vendors, and organizing IT equipment distribution required.",
        ),
        (
            "Audio/Visual (AV) Support Specialist with 3 years experience configuring conference room Zoom Rooms, Crestron control panels, wireless microphones, and digital video projectors.",
            "Conference Room AV Specialist needed. Experience setting up meeting room video conferencing, troubleshooting audio feedback, and hardware connections required.",
        ),
        (
            "Entry-Level Helpdesk Associate with hands-on customer support experience and keen interest in tech. Skilled in troubleshooting printer jams, email setup on smartphones, and browser cache issues.",
            "Junior IT Helpdesk Agent. Great entry opportunity for enthusiastic communicators with basic computer troubleshooting skills and willingness to learn.",
        ),
        (
            "Linux Helpdesk Administrator with 2 years experience supporting software engineers on Ubuntu and Fedora workstations. Managed SSH keys, package dependencies, and local Docker setups.",
            "Linux Desktop Support Specialist needed to assist engineering teams. Strong knowledge of desktop Linux distributions, command-line utilities, and package managers required.",
        ),
        (
            "Data Center Operations Technician with 2 years experience racking and cabling 1U/2U rack servers, replacing failed hard drives, and tracing fiber optic patch cables in high-density data centers.",
            "Data Center Technician wanted. Experience mounting server hardware, cable dressing, power distribution units, and hardware component swap required.",
        ),
        (
            "Junior Network Support Technician with CompTIA Network+ certification. Assisted senior engineers in patching switch ports, running cable continuity tests, and configuring basic VLAN tags.",
            "Junior Network Technician. Requirements: Understanding of IP subnets, OSI model layers, cable testing tools, and foundational networking knowledge.",
        ),
        (
            "IT Onboarding Coordinator with 2 years experience provisioning new hire user accounts across Google Workspace, Slack, Jira, and shipping pre-configured laptops.",
            "IT Identity & Onboarding Specialist. Experience managing user account creation, provisioning SaaS tool access, and delivering smooth new-hire tech experiences.",
        ),
        (
            "Software Quality Control Assistant with 1 year experience testing mobile apps on iOS and Android test devices. Verified layout consistency across various screen sizes and reported UI glitches.",
            "Mobile App QA Assistant wanted. Must have strong attention to visual detail, familiarity with iOS and Android devices, and clear bug writing skills.",
        ),
        (
            "Print Fleet & Office Tech Specialist with 3 years experience managing commercial Xerox multifunction copiers, toner replenishment, and network print queues.",
            "Office Technology Support Specialist. Responsibilities include maintaining office printers, scanners, conference room equipment, and general IT supplies.",
        ),
        (
            "Junior Webmaster / IT Assistant with experience updating WordPress page content, renewing SSL certificates, and troubleshooting DNS record changes.",
            "Web Operations & IT Assistant. Experience managing WordPress content, basic DNS concepts, and assisting staff with digital tools required.",
        ),
        (
            "Disaster Recovery & Backup IT Technician with 2 years experience verifying daily Veeam backup completion, swapping offsite backup tape cartridges, and testing restore drills.",
            "Backup Operations IT Associate. Experience monitoring automated backup jobs, error remediation, and performing test restores required.",
        ),
        (
            "Computer Repair Shop Technician with 3 years experience removing computer viruses, installing clean Windows OS copies, and replacing mechanical hard drives with SSDs.",
            "Computer Repair Technician for refurbishment depot. Experience diagnosing hardware faults, thermal paste replacement, and Windows installations required.",
        ),
        (
            "Junior IT Security & Compliance Assistant with 1 year experience reviewing employee phishing test results and auditing inactive user accounts in Active Directory.",
            "Junior IT Compliance Assistant. Experience with Active Directory user audits, basic security checklists, and documentation required.",
        ),
        (
            "IT Intern with enrollment in Computer Science degree. Provided walk-up IT bar support for university students, resolved Wi-Fi login issues, and reset student passwords.",
            "IT Helpdesk Intern. Opportunity for current students to gain hands-on customer support and enterprise technology troubleshooting experience.",
        ),
    ]

    it_negatives = [
        (
            "Commercial Roofer with 7 years experience applying hot asphalt roofs, installing rubber membrane sheeting, and sealing drain scuppers.",
            "Seeking an IT Helpdesk Support Specialist to deliver technical desktop support. Requirements: Windows/macOS, Active Directory, and ServiceNow.",
        ),
        (
            "House Painter with 5 years experience masking trim, filling nail holes with spackle, and rolling ceilings with eggshell paint.",
            "Hiring a Junior Quality Assurance (QA) Tester. Must have experience writing test scenarios, executing manual regression test passes, and Jira.",
        ),
        (
            "Drywaller with 4 years experience lifting sheetrock, cutting electrical box openings, and taping joints with mesh tape.",
            "Systems & Network Support Administrator wanted. Requirements: Experience administering office networks, DHCP/DNS, and Microsoft Intune.",
        ),
        (
            "Hairdresser with 6 years experience performing shampoo blowdrys, applying foil highlights, and trimming split ends.",
            "Entry-Level IT Technician needed. CompTIA A+ preferred. Customer service mindset, hardware knowledge, and troubleshooting skills.",
        ),
        (
            "Bartender with 3 years experience pouring draft beers, polishing glassware, and restocking beer coolers during evening shifts.",
            "IT Graduate Associate wanted. Requirements: Degree in IT/Computer Science, understanding of networking fundamentals, and Linux/Windows.",
        ),
        (
            "Bakery Assistant with 4 years experience packaging dinner rolls, kneading pie dough, and wiping down stainless steel mixing bowls.",
            "Desktop Hardware Support Specialist needed to manage laptop fleet. Experience with hardware diagnostics and OS imaging.",
        ),
        (
            "Automotive Detailer with 5 years experience buffing car clearcoats, cleaning leather seats, and spraying tire shine.",
            "Junior QA Automation Engineer. Qualifications: Basic JavaScript skills, experience writing UI tests in Cypress or Selenium.",
        ),
        (
            "Plumbing Laborer with 4 years experience carrying cast iron pipes, replacing faucet washers, and snaking shower drains.",
            "Technical Support Representative needed for software company. Must have written communication skills and ticketing experience in Zendesk.",
        ),
        (
            "Hotel Maid with 3 years experience vacuuming hotel room carpets, making king beds, and sanitizing bathroom fixtures.",
            "Field IT Deployment Technician wanted. Experience installing retail hardware, running Ethernet cabling, and POS systems.",
        ),
        (
            "Bus Driver with 6 years experience operating commuter buses, monitoring passenger fares, and adhering to strict route timetables.",
            "IT Procurement & Asset Coordinator. Experience managing computer hardware inventory, working with vendors, and asset tracking.",
        ),
        (
            "Forklift Operator with 5 years experience driving pallet jacks, loading flatbed trailers, and stacking wooden crates in warehouses.",
            "Conference Room AV Specialist needed. Experience setting up meeting room video conferencing (Zoom Rooms) and audio feedback troubleshooting.",
        ),
        (
            "Landscape Laborer with 4 years experience mowing soccer fields, weedwhacking fence lines, and raking autumn leaves.",
            "Junior IT Helpdesk Agent. Great entry opportunity for enthusiastic communicators with basic computer troubleshooting skills.",
        ),
        (
            "Pet Groomer with 3 years experience bathing golden retrievers, trimming dog nails, and brushing thick animal coats.",
            "Linux Desktop Support Specialist needed to assist engineering teams. Strong knowledge of desktop Linux distributions and command line.",
        ),
        (
            "Shoe Salesperson with 2 years experience retrieving sneaker boxes, organizing retail displays, and running register sales.",
            "Data Center Technician wanted. Experience mounting server hardware, cable dressing, and hardware component swap.",
        ),
        (
            "Carpet Installer with 7 years experience cutting carpet padding, seaming carpet joints, and tucking edges into baseboard tack strips.",
            "Junior Network Technician. Requirements: Understanding of IP subnets, OSI model layers, and cable testing tools.",
        ),
        (
            "Flight Attendant with 5 years experience delivering pre-flight safety briefings, serving tea and coffee, and calming nervous passengers.",
            "IT Identity & Onboarding Specialist. Experience managing user account creation (Google Workspace), and laptop provisioning.",
        ),
        (
            "Upholsterer with 6 years experience stapling fabric over chair frames, tying coil springs, and sewing decorative welt cord.",
            "Mobile App QA Assistant wanted. Must have strong attention to visual detail, familiarity with iOS and Android devices, and bug logging.",
        ),
        (
            "Tire Technician with 4 years experience patching tire tread punctures, balancing steel rims, and checking wheel alignment.",
            "Office Technology Support Specialist. Responsibilities include maintaining office printers, scanners, and conference room tech.",
        ),
        (
            "Commercial Window Cleaner with 5 years experience washing second-story office glass using water-fed extension poles.",
            "Web Operations & IT Assistant. Experience managing WordPress content, basic DNS concepts, and assisting staff with digital tools.",
        ),
        (
            "Deli Counter Worker with 3 years experience slicing roast beef to order, wrapping cheese wedges, and cleaning meat display cases.",
            "Backup Operations IT Associate. Experience monitoring automated backup jobs, error remediation, and performing test restores.",
        ),
        (
            "Swimming Pool Attendant with 2 years experience testing pool water alkalinity, vacuuming pool bottom sediment, and rolling pool covers.",
            "Computer Repair Technician for refurbishment depot. Experience diagnosing hardware faults, thermal paste replacement, and Windows.",
        ),
        (
            "Golf Caddy with 4 years experience carrying two golf bags per round, replacing divots, and cleaning golf irons with wet towels.",
            "IT Helpdesk Intern. Opportunity for current students to gain hands-on customer support and enterprise technology troubleshooting experience.",
        ),
    ]

    for r, j in it_positives:
        examples.append(DatasetExample(r, j, 1, "General IT & Entry-Level"))
    for r, j in it_negatives:
        examples.append(DatasetExample(r, j, 0, "General IT & Entry-Level"))

    return examples


def generate_dataset_csv(output_path: str, random_seed: int = 42) -> pd.DataFrame:
    """Generate and save the curated dataset to CSV with deterministic shuffling."""
    examples = build_curated_dataset()
    random.seed(random_seed)
    random.shuffle(examples)

    df = pd.DataFrame([
        {
            "resume_text": ex.resume_text.strip(),
            "job_description": ex.job_description.strip(),
            "label": ex.label,
            "category": ex.category,
        }
        for ex in examples
    ])

    # Ensure no exact duplicate pairs
    df = df.drop_duplicates(subset=["resume_text", "job_description"]).reset_index(drop=True)

    df.to_csv(output_path, index=False, encoding="utf-8")
    return df


if __name__ == "__main__":
    from pathlib import Path

    dest = Path(__file__).resolve().parent.parent.parent / "data" / "ml_dataset" / "expanded_dataset.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset_csv(str(dest))
    print(f"Generated {len(df)} curated examples across 8 domains.")
    print("Class distribution:")
    print(df["label"].value_counts().to_dict())
    print("\nDomain distribution:")
    print(df["category"].value_counts().to_dict())
