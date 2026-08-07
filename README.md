# Real-time Visualization Components

This repository accompanies our Real-time Visualization Alpha Lab guide.

It demonstrates Neon's [Real-time API](https://docs.pupil-labs.com/neon/real-time-api/) with different visualizations, separated into individual examples. The components are programmed in Python, Matplotlib, and PyQtGraph to focus on the core concepts and enable accessibility.

The components in this respository are:

- Full visualization of a 3D eye model, including a way to qualitatively assess vergence in real-time:

  - `realtime_eye_state.py`
  - `threeD_eye_model.py`

- A 3D representation of Neon's orientation in world-based coordinates:

  - `realtime_imu.py`

- Live update of graphs that display pupil diameter and blinks:

  - `realtime_pupil_diameter.py`
  - `realtime_blinks.py`

To try them, first make sure that your Neon and computer are connected to the same local network. Then, you can simply start an example with the `uv` command-line tool. For example:

```shell
uv run -s realtime_eye_state.py
```

If you prefer a conventional Python approach, then make a new virtual environment & activate it, install the dependencies (`pip install -r requirements.txt`), and run the examples:

```shell
python realtime_eye_state.py
```

If you are unable to stream the data, then try entering the IP address of the Companion Device in the `FALLBACK_DEVICE_ADDRESS` variable at the top of the respective Python file. Otherwise, check [our Troubleshooting guide](https://pupil-labs.github.io/pl-realtime-api/dev/troubleshooting/) and make sure to reach out to [us on Discord](https://pupil-labs.com/chat).
