# 📈 Extracción de Embeddings Financieros con Chronos Bolt

## 1. Descripción del Proyecto
Este módulo tiene como objetivo transformar series de tiempo financieras crudas (precios de **BTC** y **TSLA**) en representaciones vectoriales latentes (**embeddings**) utilizando el modelo de base **Amazon Chronos Bolt**.

El propósito es capturar la estructura profunda del mercado, la volatilidad y los patrones secuenciales en vectores densos de alta dimensionalidad (768 features), que servirán como *input* (X) para modelos predictivos posteriores.

---

## 2. Lógica de Negocio: Ventanas de Tiempo

Para maximizar la relevancia de la información capturada, se han definido ventanas de observación (*Lookback Windows*) específicas según la naturaleza del activo:

| Activo | Ventana | Justificación Financiera |
| :--- | :--- | :--- |
| **TSLA** (Tesla) | **8 Días** | Cubre una semana bursátil completa (5 días) más 3 días de contexto adicional para confirmar la tendencia de la semana anterior y suavizar el ruido de inicio de semana. |
| **BTC** (Bitcoin) | **10 Días** | Captura una visión más amplia del ciclo de mercado crypto (24/7). 10 días permiten al modelo identificar patrones de volatilidad de corto plazo y correcciones que una ventana semanal estándar podría perder. |

---

## 3. Arquitectura Técnica

El proceso utiliza **Chronos Bolt Base**, un modelo basado en la arquitectura T5. A diferencia de los modelos de lenguaje que tokenizan palabras, Chronos "parchea" (patches) la serie de tiempo.

### Flujo de Datos

```mermaid
graph LR
    A["Raw Data Parquet"] --> B{"Slicing Logic\n(Last 8 or 10 days)"}
    B --> C["Tensor (Batch, Time)"]
    C --> D["Chronos Bolt Encoder"]
    D --> E["Embeddings Latentes"]
    E --> F["Parquet File (2, 768)"]