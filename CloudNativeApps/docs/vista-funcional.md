---
layout: default
title: Vista Funcional
nav_order: 2
---

# Vista Funcional

![Vista funcional](./diagrams/networks_e2.png "Vista funcional")

## Nuevos Componentes (Entrega 3)

| Atributo | Detalle |
|---|---|
| **Componente** | Servicio de Tarjetas de Crédito |
| **Código/Id** | `credit_cards_app` |
| **Tipo** | Microservicio (Contenedor API REST) |
| **Responsabilidad** | Gestionar el almacenamiento temporal y solicitud de verificación de tarjetas de crédito de los usuarios. Recibe la petición sincrónica, guarda la tarjeta `POR_VERIFICAR` y publica un evento asíncrono. |
| **Consideraciones de diseño** | **Tradeoff:** Retorno HTTP rápido al usuario sin esperar la verificación (mejor UX) a cambio de consistencia eventual. **Punto de dolor:** Requiere infraestructura adicional de mensajería para no bloquear hilos sincrónicos. |
| **Integraciones** | Recibe HTTP POST/GET del Ingress. Publica mensajes a `credit_cards_queue` (SQS) (SDK boto3). Consulta sincrónicamente `true-native` para obtener el RUV inicial (HTTP POST). Interactúa con base de datos RDS PostgreSQL. |

| Atributo | Detalle |
|---|---|
| **Componente** | Cola de Mensajes (Event-Driven) |
| **Código/Id** | `credit_cards_queue` |
| **Tipo** | Servicio de mensajería Serverless (AWS SQS) |
| **Responsabilidad** | Desacoplar el servicio web del proceso demorado de polling de validación. Almacena temporalmente los eventos de nuevas tarjetas hasta que el poller los consuma. |
| **Consideraciones de diseño** | Garantiza la entrega del mensaje y tolerancia a fallos. Si el worker cae por un timeout de TrueNative, el mensaje se reintenta automáticamente. |
| **Integraciones** | Integración asíncrona (pub/sub). Recibe de `credit_cards_app`, consumido (trigger) por `aws_async_poller`. |

| Atributo | Detalle |
|---|---|
| **Componente** | Worker Poller Asíncrono |
| **Código/Id** | `aws_async_poller` |
| **Tipo** | Función Serverless (AWS Lambda) |
| **Responsabilidad** | Ejecutar el polling constante contra TrueNative para saber el estado de la verificación, actualizar la base de datos de Tarjetas y notificar al usuario. |
| **Consideraciones de diseño** | **Tradeoff:** Ejecución aislada que no consume recursos del cluster de EKS directamente. Requiere VPC Access para llegar a RDS. Garantiza cumplir la restricción de un solo hilo asíncrono sin afectar a la aplicación principal FastAPI. |
| **Integraciones** | Lee de `credit_cards_queue` (Trigger Evento). Consulta GET iterativo a `true_native` expuesto vía el Ingress Público. Actualiza de manera directa `dann-db` mediante TCP (psycopg2). Envía correos de confirmación vía AWS SES (SDK boto3). |

| Atributo | Detalle |
|---|---|
| **Componente** | Proveedor de Identidad y Pagos MOCK |
| **Código/Id** | `true_native` |
| **Tipo** | Servicio Externo Simulado (Contenedor K8s) |
| **Responsabilidad** | Simular la verificación demorada (latencia de 30 a 120seg) de identidad y fraude de tarjetas de crédito. |
| **Consideraciones de diseño** | Desplegado dentro del mismo cluster para facilidad administrativa, pero expuesto y consumido exclusivamente por su URL pública (ALB vía Ingress) para simular rigurosamente la integración con la API de un tercero en internet. |
| **Integraciones** | Expone API REST pública (POST/GET/PATCH). Accesible por `credit_cards_app` y `aws_async_poller`. |
