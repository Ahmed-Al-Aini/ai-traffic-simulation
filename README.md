# Traffic Simulation

## التشغيل

```bash
pip install -r requirements.txt
python main.py
```

## البنية

- `traffic_sim/config.py`: ثوابت الإعداد.
- `traffic_sim/models.py`: التعدادات ونماذج البيانات.
- `traffic_sim/emergency.py`: المؤسسات والمركبات الطارئة.
- `traffic_sim/controller.py`: وحدة التحكم في الإشارات والأولوية.
- `traffic_sim/ai.py`: نموذج DQN والوكيل الذكي.
- `traffic_sim/simulator.py`: المحاكاة والرسم وحلقة التشغيل.
- `main.py`: نقطة الدخول.