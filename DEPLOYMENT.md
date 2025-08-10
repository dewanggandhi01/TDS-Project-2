# 🚀 Deployment Guide

This guide covers deploying the LLM Data Analyst Agent to various platforms.

## 📋 Pre-deployment Checklist

- [ ] OpenAI API key ready
- [ ] Project built and tested locally
- [ ] Repository pushed to GitHub
- [ ] Environment variables documented

## 🌟 Vercel Deployment (Recommended)

### One-Click Deploy
1. Click: [![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/dewanggandhi01/TDS-Project-2)
2. Connect your GitHub account
3. Set environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `ENVIRONMENT`: `production`
4. Deploy!

### Manual Deployment
```bash
# Install Vercel CLI
npm install -g vercel

# Login to Vercel
vercel login

# Deploy
vercel

# Set environment variables
vercel env add OPENAI_API_KEY
vercel env add ENVIRONMENT production

# Deploy to production
vercel --prod
```

### Environment Variables in Vercel
1. Go to your project dashboard on Vercel
2. Navigate to Settings → Environment Variables
3. Add these variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `ENVIRONMENT`: `production`
   - `OPENAI_MODEL`: `gpt-4` (optional)
   - `OPENAI_MAX_TOKENS`: `2000` (optional)

## 🚄 Railway Deployment

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Set environment variables
railway variables set OPENAI_API_KEY=your_key_here
railway variables set ENVIRONMENT=production

# Deploy
railway up
```

## 🟣 Heroku Deployment

```bash
# Install Heroku CLI
# Create new app
heroku create your-app-name

# Set environment variables
heroku config:set OPENAI_API_KEY=your_key_here
heroku config:set ENVIRONMENT=production

# Deploy
git push heroku main
```

## 🎨 Render Deployment

1. Go to [render.com](https://render.com)
2. Connect your GitHub repository
3. Create a new **Web Service**
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python run_server.py`
5. Add environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `ENVIRONMENT`: `production`
6. Deploy

## ☁️ Google Cloud Run

```bash
# Build and deploy
gcloud run deploy llm-data-analyst \
  --source . \
  --platform managed \
  --region us-central1 \
  --set-env-vars OPENAI_API_KEY=your_key_here,ENVIRONMENT=production
```

## 🐳 Docker Deployment

```dockerfile
# Create Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "run_server.py"]
```

```bash
# Build and run
docker build -t llm-data-analyst .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_key_here \
  -e ENVIRONMENT=production \
  llm-data-analyst
```

## 🔧 Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ | - | Your OpenAI API key |
| `ENVIRONMENT` | ✅ | - | Set to `production` |
| `OPENAI_MODEL` | ❌ | `gpt-4` | OpenAI model to use |
| `OPENAI_MAX_TOKENS` | ❌ | `2000` | Max tokens for code generation |
| `OPENAI_TEMPERATURE` | ❌ | `0.1` | Creativity level (0-1) |
| `PORT` | ❌ | `8000` | Server port |
| `HOST` | ❌ | `0.0.0.0` | Server host |
| `LOG_LEVEL` | ❌ | `info` | Logging level |

## 🧪 Post-Deployment Testing

After deployment, test these endpoints:

```bash
# Health check
curl https://your-app.vercel.app/health

# API health
curl https://your-app.vercel.app/api/health

# Test LLM analysis
curl -X POST "https://your-app.vercel.app/api/llm/analyze" \
  -H "Content-Type: application/json" \
  -d '{"task": "Count rows in dataset"}'
```

## 🔍 Troubleshooting

### Common Issues

**1. OpenAI API Key Not Working**
- Verify the key is correct
- Check if you have credits in your OpenAI account
- Ensure the key has the correct permissions

**2. Build Failures**
- Check if all dependencies are in `requirements.txt`
- Verify Python version compatibility
- Look at build logs for specific errors

**3. Import Errors**
- Ensure `PYTHONPATH` is set correctly
- Check that all modules are properly structured
- Verify `api/index.py` exists and imports correctly

**4. Timeout Issues**
- LLM analysis can take 30+ seconds
- Increase timeout limits if possible
- Consider splitting complex queries

### Getting Help

1. Check the [GitHub Issues](https://github.com/dewanggandhi01/TDS-Project-2/issues)
2. Review platform-specific documentation
3. Test locally first before deploying
4. Check application logs for detailed error messages

## 📊 Monitoring

### Recommended Monitoring
- **Uptime**: Use tools like UptimeRobot
- **Performance**: Monitor response times
- **API Usage**: Track OpenAI API usage and costs
- **Error Rates**: Monitor 4xx and 5xx responses

### Log Analysis
Most platforms provide built-in logging. Monitor for:
- OpenAI API errors
- Python import errors
- Memory usage issues
- Request timeout errors

---

**🎉 Once deployed, share your live demo URL and start analyzing data with AI!**
