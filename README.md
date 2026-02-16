# 📈 Extracción de Embeddings Financieros con Chronos Bolt

## 1. Descripción del Proyecto
Este módulo tiene como objetivo transformar series de tiempo financieras crudas (precios de **BTC** y **TSLA**) en representaciones vectoriales latentes (**embeddings**) utilizando el modelo de base **Amazon Chronos Bolt**.

El propósito es capturar la estructura profunda del mercado, la volatilidad y los patrones secuenciales en vectores densos de alta dimensionalidad (768 features), que servirán como *input* (X) para modelos predictivos posteriores, evitando el uso de precios explícitos para reducir el ruido.

---

## 2. Lógica de Negocio: Ventanas de Tiempo

Para maximizar la relevancia de la información capturada, se han definido ventanas de observación (*Lookback Windows*) específicas según la naturaleza del activo:

| Activo | Ventana | Justificación Financiera |
| :--- | :--- | :--- |
| **TSLA** (Tesla) | **5 Días** | Corresponde a una **semana bursátil estándar** (Lunes a Viernes). Al excluir fines de semana (donde no hay mercado), capturamos la "vela semanal" pura sin ruido de huecos temporales. |
| **BTC** (Bitcoin) | **8 Días** | Bitcoin opera 24/7. Una ventana de 8 días captura un **ciclo semanal completo (7 días)** más el día de confirmación (*momentum*) respecto al mismo día de la semana anterior. |

---

## 3. Arquitectura Técnica

El proceso utiliza **Chronos Bolt Base**, un modelo basado en la arquitectura T5. A diferencia de los modelos de lenguaje que tokenizan palabras, Chronos "parchea" (patches) la serie de tiempo.

### Flujo de Datos

```mermaid
graph LR
    A[Raw Data Parquet] --> B{Slicing Logic\n(Last 5 or 8 days)}
    B --> C[Tensor (Batch, Time)]
    C --> D[Chronos Bolt Encoder]
    D --> E[Embeddings Latentes]
    E --> F[Parquet File (2, 768) o Parquet File (2, 512) dependiendo si es base o small]