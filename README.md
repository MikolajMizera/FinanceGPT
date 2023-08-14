# FinanceGPT
GPT model to support investment market research.

## Description
This repository contains the implementation of a system that enables fine-tuning and inference of Large Language Models (LLMs) using textual and OHLC (Open, High, Low, Close) data. This project utilizes various APIs for data fetching and the FastAPI framework for serving the model. The system supports integration with MongoDB for storing and retrieving datasets, and the Docker for containerization. The project also plans for future scalability through Kubernetes orchestration.

## Getting Started

### Dependencies
- Python 3.8 or later
- FastAPI
- Pydantic
- MongoDB
- Hydra
- Docker

### Installation
1. Clone the repository
2. Install the package along with dependencies with `pip install .`

### Usage
...

### **Core Components**
The core components above represent foundational design principles or blueprints 
guiding the project's architecture. They aren't rigid structures, but flexible 
guidelines ensuring consistent, high-quality implementations.

1. **Data Adapters**:
   - Unified interfaces like **TextApiDataAdapter** and **OhlcApiDataAdapter** power seamless integrations with diverse data sources, including specialized adapters like **RedditTextApiDataAdapter** and **YahooOhlcApiDataAdapter**.

2. **Datasets**:
   - Robust collections encapsulating textual or numerical **DataPoints**, serving as the foundational building blocks for data processing.

3. **Database Connectors**:
   - Advanced modules driving efficient database management, error-handling, and end-to-end data integrity.

4. **Prompt Factory**:
   - An innovative system transforming raw textual and numerical data into actionable prompts for cutting-edge LLMs.

5. **Model Interface**:
   - A versatile facade enabling integration with a broad spectrum of ML models through a unified interface.

6. **Access Management**:
   - Granular user roles (`Admin`, `User`, and `PremiumUser`) underpin a secure and organized access structure.

### **Deployment Strategy**
- Leveraging Docker, the MVP assures streamlined deployments, be it on local infrastructures or GPU-empowered remote servers.
- Secure storage of Docker images in Docker Hub ensures quick access and distribution.
- Architectured for the future, our system is primed for elastic scaling in the cloud, especially using Ray Clusters.

### **Development Tools & Practices**
- **FastAPI** anchors our backend, offering robust API development combined with state-of-the-art user management.
- Continuous Integration gets a boost with **GitHub Actions**, automating tests and code checks for every push or pull request.
- A combination of **DVC** and **Git** introduces version-controlled code and database schema, promoting reproducibility.
- Our partnership with **S3 Bucket** not only provides expansive storage for datasets but also integrates seamlessly with DVC for data versioning.

### **Database**
- Relying on MongoDB, we offer a dynamic datastore tailored to modern applications.
- A suite of collections, including but not limited to `users`, `datasets`, and `inferenceResults`, organizes data efficiently.

### **Security**
- Trust is paramount. With FastAPI's innate OAuth2.0 capabilities, we ensure top-notch security protocols.
- A meticulous user role system fortifies role-based access, preventing unauthorized breaches.

### **Development & Maintenance**
- Trust Anaconda to maintain the sanctity of our Python environments and dependencies.
- Guided by the `pyproject.toml` configuration, the project architecture remains modular and extensible.
- Embracing Docker's capabilities, we guarantee consistent environments, whether during development or production.
- Upholding coding standards, we employ a toolchain featuring `flake8`, `mypy`, `black`, and `pytest` to ensure code quality.

## Project Roadmap

### **Phase 1: Project Setup and Foundations**
1. **Environment Setup**: Initialize Anaconda environment and integrate essential tools like FastAPI, MongoDB, and DVC.
2. **Database Configuration**: Set up MongoDB, create initial schemas, and define primary collections.
3. **API Development**: Lay down foundational routes and endpoints using FastAPI.

### **Phase 2: Core Component Implementation**
4. **Data Adapters**:
   - Develop and test primary data adapters.
   - Incorporate specialized adapters like **RedditTextApiDataAdapter**.
5. **Datasets**:
   - Implement base dataset structures.
   - Add and test specialized dataset structures like **OhlcDataset**.
6. **Database Connectors**: Implement database storing, loading, and error-handling functionalities.
7. **Prompt Factory**: Initiate transformation logics for creating prompts.

### **Phase 3: Model Integration and Testing**
8. **Model Interface Development**: Incorporate various ML models and test their integrations.
9. **Training and Fine-tuning**: Use the **TrainingPromptCollection** to train and fine-tune models.
10. **Inference Testing**: Validate model predictions using **InferencePromptCollection**.

### **Phase 4: User Management and Security**
11. **User Role Implementation**: Define and implement user roles (`Admin`, `User`, and `PremiumUser`).
12. **OAuth2.0 Integration**: Set up OAuth2.0 with FastAPI for authentication.
13. **Access Controls**: Implement role-based access controls for various API endpoints.

### **Phase 5: Scalability and Deployment**
14. **Dockerization**: Create a Dockerfile and ensure reproducible builds.
15. **Cloud Integration**: Lay foundations for cloud scalability using AWS, GCP, etc.
16. **Continuous Deployment**: Integrate with GitHub Actions for automated deployments.

### **Phase 6: Continuous Improvement and Iterations**
17. **User Feedback Collection**: Incorporate mechanisms to collect user feedback.
18. **Performance Optimization**: Analyze system performance and optimize bottlenecks.
19. **Additional Data Sources**: Integrate more data adapters based on user demand and industry trends.

### **Phase 7: Premium Features and Monetization**
20. **Advanced Analysis Tools**: Develop and roll out tools and features specific to `PremiumUser`.
21. **Monetization Strategy**: Explore subscription models, pay-per-use, or other monetization strategies.

### **Phase 8: Maintenance and Future Considerations**
22. **Regular Security Audits**: Ensure system remains secure with evolving security standards.
23. **Tech Stack Upgrades**: Regularly update tools and frameworks used in the project.
24. **Expansion Considerations**: Explore the potential for integrating other financial or AI tools based on market demands.

---

You can customize this roadmap based on specific project needs, expected timelines, and team capabilities. It's essential to maintain flexibility and adapt the roadmap based on project progress and any new requirements or constraints that emerge.

## Contributing
This project currently doesn't accept outside contributions.

## License
...
