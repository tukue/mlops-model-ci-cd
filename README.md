# MLOps Model CI/CD Pipeline

A production-ready MLOps system demonstrating end-to-end machine learning lifecycle management with automated CI/CD, model versioning, and monitoring.

## 🚀 Features

- **Automated ML Pipeline**: GitHub Actions CI/CD with model training and testing
- **Model Versioning**: Custom registry with deployment logic and rollback capability
- **Production API**: FastAPI with health checks, metrics, and error handling
- **Monitoring**: Prometheus metrics for predictions, latency, and API requests
- **Containerization**: Docker support for consistent deployment
- **Testing**: Comprehensive unit and integration tests

## 🏗️ Architecture

```
GitHub → Actions → Docker → Production API
   ↓         ↓        ↓         ↓
  Code → Test → Build → Deploy → Monitor
```

## 📊 API Endpoints

- `GET /` - API information and available endpoints
- `GET /health` - Health check for load balancers
- `POST /predict` - Make predictions with model versioning
- `GET /model-info` - Active model metadata and metrics
- `GET /metrics` - Prometheus metrics for monitoring
- `GET /docs` - Interactive API documentation

## 🛠️ Quick Start

### Prerequisites
- Python 3.9+
- Docker (optional)

### Local Development

1. **Clone and setup**
   ```bash
   git clone <repo-url>
   cd mlops-model-ci-cd
   pip install -r requirements.txt
   ```

2. **Train model**
   ```bash
   python -m src.train
   ```

3. **Run tests**
   ```bash
   pytest tests/ -v
   ```

4. **Start API**
   ```bash
   uvicorn app.main:app --reload
   ```

5. **Test prediction**
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
   ```

### Docker Deployment

```bash
# Build and run
docker-compose up --build

# Or manually
docker build -t mlops-api .
docker run -p 8000:8000 mlops-api
```

## 📈 Model Registry

The system includes a custom model registry that:

- **Versions models** with timestamps
- **Tracks metrics** (accuracy, test samples)
- **Manages deployments** with automatic promotion
- **Enables rollbacks** to previous versions
- **Cross-platform compatibility** (Linux symlinks, Windows copy)

### Model Deployment Logic
- Models with accuracy > 90% are automatically deployed
- Previous models remain available for rollback
- Metadata stored in JSON format for easy inspection

## 🔍 Monitoring & Observability

### Prometheus Metrics
- `ml_predictions_total` - Total predictions made
- `ml_prediction_duration_seconds` - Prediction latency
- `api_requests_total` - API requests by method/endpoint/status

### Health Monitoring
- `/health` endpoint for uptime checks
- Structured error handling with proper HTTP codes
- Request tracking middleware

## 🧪 Testing Strategy

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end API testing
- **Model Tests**: Training pipeline validation
- **CI/CD Tests**: Docker build and endpoint verification

## 🔄 CI/CD Pipeline

GitHub Actions workflow includes:

1. **Setup**: Python environment and dependencies
2. **Train**: Model training with registry
3. **Test**: Unit and integration tests
4. **Build**: Docker image creation
5. **Validate**: Live API endpoint testing

## 📁 Project Structure

```
├── app/
│   ├── main.py          # FastAPI application
│   └── schemas.py       # Pydantic models
├── src/
│   ├── train.py         # Model training script
│   └── model_registry.py # Model versioning system
├── tests/
│   ├── test_app.py      # API tests
│   └── test_model.py    # Model tests
├── artifacts/           # Model storage
├── .github/workflows/   # CI/CD configuration
├── Dockerfile           # Container configuration
├── docker-compose.yml   # Local deployment
└── requirements.txt     # Dependencies
```

## 🎯 Production Considerations

### Scalability
- Stateless API design for horizontal scaling
- Model caching for improved performance
- Async-ready FastAPI framework

### Security
- Input validation with Pydantic schemas
- Proper error handling without information leakage
- Health checks for monitoring systems

### Reliability
- Graceful degradation when model unavailable
- Comprehensive logging for debugging
- Cross-platform compatibility

## 📊 Model Performance

- **Dataset**: Iris classification (150 samples, 4 features)
- **Algorithm**: Logistic Regression
- **Accuracy**: 96.7% on test set
- **Latency**: <100ms prediction time

## 🚀 Deployment Options

### Free Tier Deployment
- **Railway/Render**: 500 hours/month free
- **Docker Hub**: Unlimited public repositories
- **GitHub Actions**: 2000 minutes/month

### Production Deployment
- **AWS ECS/Fargate**: Container orchestration
- **Kubernetes**: Advanced scaling and management
- **Load Balancer**: High availability setup

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics
- **Model Info**: http://localhost:8000/model-info