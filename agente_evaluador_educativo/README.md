# Agente Evaluador Educativo

API en Python + FastAPI para revisar la claridad, exactitud conceptual y advertencias de
material educativo sobre opciones CALL. No recomienda comprar o vender ni ofrece estrategias.

## Ejecutar localmente

1. Instala Python 3.11.
2. Crea y activa un entorno virtual.
3. Instala dependencias:

```bash
pip install -r requirements.txt
```

4. Configura `OPENAI_API_KEY` como variable de entorno. No subas la clave a GitHub.
5. Ejecuta:

```bash
uvicorn main:app --reload
```

6. Abre `http://127.0.0.1:8000/docs`.

## Probar con texto

En `POST /evaluate`, envía:

```json
{
  "text": "Pega aquí el contenido educativo que deseas evaluar."
}
```

## Probar con URL

```json
{
  "url": "https://opciones-call-equipo9.streamlit.app/"
}
```

La extracción automática solo intenta consultar ese dominio y no sigue redirecciones.
Si Streamlit solicita autenticación, copia el texto visible de la aplicación y usa el campo
`text`. Las páginas renderizadas con JavaScript también pueden requerir pegar el contenido.

## Desplegar en Render

1. Sube estos archivos a un repositorio de GitHub.
2. En Render, selecciona **New + > Blueprint** y conecta el repositorio que contiene `render.yaml`.
   Alternativamente, crea un **Web Service** manual con:
   - Runtime: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. En Environment, crea `OPENAI_API_KEY` con tu clave privada.
4. Despliega y abre `https://TU-SERVICIO.onrender.com/docs`.

## Endpoints

- `GET /`: estado general
- `GET /health`: comprobación de salud
- `POST /evaluate`: evaluación educativa

## Seguridad y límites

- La clave de API se guarda como variable de entorno, nunca en el código.
- La extracción de URL está restringida al dominio del proyecto.
- El contenido recibido se considera material no confiable para analizar.
- El resultado es orientativo y requiere revisión humana.
- El plan gratuito de Render puede suspender servicios inactivos y provocar demora al primer acceso.
