# Inicio Rápido - Sistema A2A con Groq

## 🚀 Inicio en 3 Pasos

### 1. Verificar Variables de Entorno

Asegúrate de que tu archivo `.env` en el root del proyecto tenga:

```bash
GROQ_API_KEY=tu_groq_api_key_aqui
GOOGLE_API_KEY=tu_google_api_key_aqui  # Solo para generación de imágenes
TAVILY_API_KEY=tu_tavily_api_key_aqui
```

### 2. Iniciar Todos los Servicios

```bash
./restart-all.sh
```

Este script:
- ✅ Detiene todos los procesos anteriores
- ✅ Inicia el backend orchestrator
- ✅ Inicia los 3 agentes (Medical, Multimodal, Images)
- ✅ Inicia el frontend
- ✅ Muestra los PIDs de todos los procesos

### 3. Abrir la Aplicación

Abre tu navegador en: **http://localhost:3000**

---

## 📊 Ver Logs en Tiempo Real

```bash
# Backend
tail -f logs/backend.log

# Agentes
tail -f logs/medical.log
tail -f logs/multimodal.log
tail -f logs/images.log

# Frontend
tail -f logs/frontend.log
```

---

## 🛑 Detener Todos los Servicios

```bash
pkill -f "uv run python"
pkill -f "npm run dev"
```

---

## ✅ Verificar que Todo Funciona

### Verificar Imports
```bash
cd demo/ui && uv run python -c "from service.server.beeai_host_manager import BeeAIHostManager; print('✅ Backend OK')"
cd samples/python/agents/medical_Images && uv run python -c "from app.agent import MedicalAgent; print('✅ Medical OK')"
cd samples/python/agents/multimodal && uv run python -c "from app.agent import PhysicsMultimodalAgent; print('✅ Multimodal OK')"
cd samples/python/agents/images && uv run python -c "from app.agent import ImageGenerationAgent; print('✅ Images OK')"
```

### Verificar Servicios Activos
```bash
# Ver procesos Python
ps aux | grep "uv run python"

# Ver proceso frontend
ps aux | grep "npm run dev"
```

---

## 🔧 Solución de Problemas Comunes

### Error: "Invalid API Key"
- Verifica que `GROQ_API_KEY` esté en el `.env`
- Reinicia el backend: `./restart-all.sh`

### Error: "Module not found"
- Reinstala dependencias:
  ```bash
  cd demo/ui && uv sync
  cd samples/python/agents/medical_Images && uv sync
  cd samples/python/agents/multimodal && uv sync
  cd samples/python/agents/images && uv sync
  ```

### Puerto ya en uso
- Detén todos los servicios: `pkill -f "uv run python" && pkill -f "npm run dev"`
- Espera 2 segundos
- Reinicia: `./restart-all.sh`

### Agente no responde
- Verifica que el agente esté corriendo: `ps aux | grep "python -m app"`
- Verifica los logs: `tail -f logs/[agente].log`
- Reinicia el agente específico

---

## 📚 Documentación Completa

- `RESUMEN-COMPLETO.md` - Resumen de todos los cambios
- `CAMBIO-A-GROQ.md` - Detalles de la migración a Groq
- `SOLUCION-ENV.md` - Centralización de variables de entorno
- `BACKEND-MESSAGE-FIX.md` - Fix de parsing de mensajes
- `FRONTEND-FIX.md` - Fix de respuestas API

---

## 🎯 URLs del Sistema

| Servicio | URL |
|----------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:12000 |
| Medical Agent | http://localhost:10001 |
| Multimodal Agent | http://localhost:10002 |
| Images Agent | http://localhost:10003 |

---

## 💡 Consejos

1. **Usa el script de reinicio**: `./restart-all.sh` es la forma más fácil de iniciar todo
2. **Monitorea los logs**: Los logs están en `logs/` y son útiles para debugging
3. **Verifica el .env**: Todas las API keys deben estar en el `.env` del root
4. **Groq es rápido**: Las respuestas deberían ser mucho más rápidas que con Gemini

---

## ✨ Características del Sistema

- 🤖 **3 Agentes Especializados**: Médico, Multimodal, Generación de Imágenes
- ⚡ **Groq (Llama 4)**: Respuestas ultra-rápidas
- 🎨 **Frontend React**: Interfaz moderna con Next.js
- 🔄 **Orchestrator Inteligente**: Enruta automáticamente a los agentes
- 📊 **Admin Dashboard**: Gestión completa del sistema

---

¡Listo! Tu sistema A2A está configurado y listo para usar. 🎉
