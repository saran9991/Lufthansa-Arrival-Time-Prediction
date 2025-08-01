# **H3‑Cell‑Based Arrival‑Time Prediction for Lufthansa Flights**

> LMU‑Munich Data Science Practical in cooperation with Lufthansa Group

<h2 align="center"></h2>

<p align="center">
  <img src="./src/readme_utils/h3_res.png"              width="45%" alt="H3 resolution along trajectory">
  <img src="./src/readme_utils/patterns_near_airport.png" width="53%" alt="Flight patterns near FRA">
</p>

---

## Overview

This project builds a full pipeline that turns raw ADS‑B surveillance data into minute‑level ETA predictions for Lufthansa flights inbound to Frankfurt (EDDF).
Key points:

* Uses **H3 hexagonal spatial indexing** to capture local traffic density along each trajectory
* Supports whole‑route and last‑100 km prediction modes
* Codebase: data download, feature engineering, model training, evaluation, and inference



## Feature set & modelling

* **Features:**
  * Position (distance‑to‑runway, sine/cos‑encoded lat / lon and  bearing)
  * Kinematics (altitude, vertical speed, ground speed)
  * H3 traffic density over the past 10 / 30 / 60 minutes
  * Calendar & cyclic time (weekday, holiday, sine/cos‑encoded time‑of‑day and day‑of‑year)
* **Targets:** seconds‑to‑touchdown (full) or seconds‑to‑touchdown within 100 km
* **Models:** polynomial regression, XGBoost, MLP, LSTM
* **Interpretability:** SHAP plots available in `src/evaluations/`

## License

MIT - see [LICENSE](LICENSE).

## Acknowledgements

* **Dr. Viktor Bengs** (LMU Chair of Artificial Intelligence and Machine Learning) - academic supervision and guidance
* **Dr. Sebastian Weber** - industry mentor (Lufthansa Group)
* **OpenSky Network** for ADS‑B data



