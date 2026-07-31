# Cloud Intelligence Platform for Food Supply Chain Loss Prediction and Optimization

**Course:** BITE412L - Cloud Computing  

---

## 👥 Team Members

| Name | Register Number |
| :--- | :--- |
| **Arunima** | `23BIT0099` |
| **Jacob J M** | `23BIT0238` |
| **Afthab O N** | `23BIT0304` |

---

## 📌 Problem Statement

Food loss and waste remain a critical global issue across agri-food supply chains, largely caused by:
- Inefficient inventory management and forecasting.
- Sub-optimal storage conditions (temperature, humidity, pest/rodent infestation).
- Transportation delays and lack of real-time supply chain visibility.
- Absence of unified real-time monitoring and predictive capabilities.

Existing research relies heavily on static, synthetic, or self-reported survey data, focuses on isolated stages of the supply chain, and lacks an end-to-end cloud platform capable of continuous monitoring, adaptive prediction, and automated decision support.

---

## 🎯 Objectives

1. **Predictive Analytics:** Develop machine learning models to forecast food loss percentage, quality degradation, and supply chain bottlenecks across various commodities and supply chain stages.
2. **Real-time IoT & Data Ingestion:** Establish continuous, real-time sensing for environmental parameters (temperature, humidity, transit duration, pest presence) during storage and transportation.
3. **Automated Decision Support:** Provide actionable recommendations such as dynamic inventory redistribution, route optimization, and storage condition adjustments to minimize spoilage.
4. **Cloud-Native Architecture:** Build a scalable, event-driven, serverless cloud platform (AWS / Azure) for low-latency ingestion, automated model inference, and real-time visualization.

---

## 🏗️ Proposed Architecture & Framework

The platform follows a multi-tier cloud architecture:

1. **IoT Sensing & Ingestion Layer:**
   - Real-time environmental and logistics telemetry captured by IoT edge devices.
   - Ingested via **AWS IoT Core / Azure IoT Hub** with MQTT/HTTPS protocols.
2. **Data Storage & Data Lake Layer:**
   - Raw and historical data stored in **AWS S3 / Azure Blob Storage**.
   - Structured analytical data stored in **AWS Redshift / Azure SQL Database**.
3. **Event-Driven Processing Layer:**
   - **AWS Lambda / Azure Functions** handle event triggers, data transformations, and payload validation.
4. **Machine Learning Pipeline:**
   - **AWS SageMaker / Azure Machine Learning** trains and deploys predictive regression/classification models.
   - Continuous inference computes real-time risk scores and estimated loss percentages.
5. **Analytics & Dashboard Layer:**
   - Interactive web interface displaying live metrics, supply chain heatmaps, predictive alerts, and mitigation recommendations.

*(Architecture diagrams can be found under the [`Architecture/`](./Architecture) directory)*

---

## 🛠️ Technology Stack

- **Cloud Infrastructure:** AWS (S3, Lambda, IoT Core, SageMaker, Redshift) / Microsoft Azure (Azure IoT Hub, Blob Storage, Azure Functions, Azure ML, Azure SQL)
- **Machine Learning & Analytics:** Python, Scikit-learn, Pandas, NumPy
- **IoT & Hardware Simulation:** Telemetry sensors (Temperature, Humidity, Rodents/Pest tracking, GPS)
- **Database Systems:** Relational & Data Warehouse (Azure SQL / PostgreSQL / AWS Redshift)
- **Frontend / Visualization:** Interactive Web Dashboard (Streamlit / React / HTML5)
- **Version Control:** Git & GitHub

---

## 📊 Dataset Details

The repository includes real-world historical food loss datasets located under the [`Dataset/`](./Dataset) folder.

- **Primary File:** [`Dataset/data.csv`](./Dataset/data.csv) (over 30,000+ records)
- **Data Source:** FAO (Food and Agriculture Organization), APHLIS (African Postharvest Losses Information System), and academic literature.

### Key Attributes:
- **`country` & `region`**: Geographical location of the supply chain data.
- **`commodity` & `cpc_code`**: Agricultural product classification (e.g., Wheat, Rice, Maize, Sorghum, Barley, Oats).
- **`year`**: Temporal tracking for historical trends.
- **`loss_percentage` & `loss_quantity`**: Quantified loss percentage and absolute volume lost.
- **`activity` & `food_supply_stage`**: Stage in the food value chain (e.g., Harvesting, Storage, Processing, Transportation, Household consumption).
- **`cause_of_loss`**: Drivers behind loss (e.g., Rodents, Mycotoxins, Temperature degradation, Mechanical damage).
- **`method_data_collection` & `reference`**: Primary methodology (Survey, Modelled Estimates, Controlled Experiments).

---

## 📂 Repository Structure

```
├── Architecture/        # AWS / Cloud Architecture diagrams
├── Dataset/             # Historical Food Loss & Waste datasets (data.csv)
├── Docs/                # Project documentation and review papers
├── Literature Survey/   # Collected academic papers and reference research
├── Presentation/        # Project presentation slides
├── References/          # Supporting reference material
├── Results/             # Model evaluation results and benchmarks
├── src/                 # Source code (ML models, cloud functions, dashboard)
└── README.md            # Project overview and documentation
```
