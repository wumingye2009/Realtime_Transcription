from __future__ import annotations

import re
from typing import Any

import sounddevice as sd

from app.models.audio import AudioDevice, AvailableDevicesResponse


class DeviceDiscoveryService:
    """Lists local audio devices with a conservative Windows-friendly split."""

    _OUTPUT_HOST_PRIORITY = {
        "Windows WASAPI": 0,
        "Windows DirectSound": 1,
        "MME": 2,
        "Windows WDM-KS": 3,
    }

    _MIC_HOST_PRIORITY = {
        "Windows WASAPI": 0,
        "Windows DirectSound": 1,
        "MME": 2,
        "Windows WDM-KS": 3,
    }

    def list_devices(self) -> AvailableDevicesResponse:
        warnings: list[str] = []

        try:
            devices = sd.query_devices()
            default_input, default_output = sd.default.device
            hostapis = sd.query_hostapis()
        except Exception as exc:  # pragma: no cover - depends on host audio stack
            warnings.append(f"Audio device discovery is unavailable: {exc}")
            return AvailableDevicesResponse(
                system_output_devices=[],
                microphone_input_devices=[],
                warnings=warnings,
            )

        output_candidates: list[AudioDevice] = []
        input_candidates: list[AudioDevice] = []

        for index, raw_device in enumerate(devices):
            device = self._to_audio_device(index, raw_device, hostapis)
            if raw_device["max_output_channels"] > 0:
                output_candidates.append(
                    device.model_copy(
                        update={
                            "kind": "output",
                            "is_default": index == default_output,
                            "capture_source": "system_loopback",
                            "capture_provider": "windows_loopback",
                            "supports_loopback": True,
                        }
                    )
                )
            if raw_device["max_input_channels"] > 0:
                input_candidates.append(
                    device.model_copy(
                        update={
                            "kind": "input",
                            "is_default": index == default_input,
                            "capture_source": "microphone",
                            "capture_provider": "microphone",
                            "supports_loopback": False,
                        }
                    )
                )

        output_devices = self._select_output_devices(output_candidates, default_output)
        input_devices = self._select_input_devices(input_candidates, default_input)

        if output_devices:
            warnings.append(
                "Choose the playback device your audio is actually coming out of. "
                "For wired headphones, pick the headphone / 2nd output device instead of generic Sound Mapper aliases."
            )

        return AvailableDevicesResponse(
            system_output_devices=output_devices,
            microphone_input_devices=input_devices,
            warnings=warnings,
        )

    def get_device_metadata(
        self,
        system_output_device_id: str,
        microphone_input_device_id: str | None,
    ) -> dict[str, str]:
        devices = self.list_devices()
        output_device = self._find_device(devices.system_output_devices, system_output_device_id)
        microphone_device = self._find_device(devices.microphone_input_devices, microphone_input_device_id)

        return {
            "system_output_device": self._format_device(output_device, system_output_device_id),
            "microphone_input_device": self._format_device(microphone_device, microphone_input_device_id),
        }

    @staticmethod
    def _find_device(devices: list[AudioDevice], device_id: str | None) -> AudioDevice | None:
        if not device_id:
            return None

        for device in devices:
            if device.id == device_id:
                return device
        return None

    @staticmethod
    def _format_device(device: AudioDevice | None, fallback_id: str | None) -> str:
        if device is None:
            if fallback_id:
                return f"Unknown device (ID {fallback_id})"
            return "None"

        hostapi = f", host API {device.hostapi}" if device.hostapi else ""
        provider = f", provider {device.capture_provider}" if device.capture_provider else ""
        return f"{device.name} (ID {device.id}{hostapi}{provider})"

    @staticmethod
    def _to_audio_device(index: int, raw_device: Any, hostapis: list[dict[str, Any]]) -> AudioDevice:
        hostapi_index = raw_device.get("hostapi")
        hostapi_name = None
        if isinstance(hostapi_index, int) and 0 <= hostapi_index < len(hostapis):
            hostapi_name = hostapis[hostapi_index].get("name")

        return AudioDevice(
            id=str(index),
            name=raw_device["name"],
            description=None,
            kind="output",
            channels=max(raw_device["max_input_channels"], raw_device["max_output_channels"]),
            default_samplerate=raw_device.get("default_samplerate"),
            hostapi=hostapi_name,
        )

    def _select_output_devices(self, devices: list[AudioDevice], default_output: int | None) -> list[AudioDevice]:
        filtered = [device for device in devices if self._is_meaningful_output_device(device)]
        deduped = self._dedupe_devices(filtered, kind="output")
        return [self._decorate_device(device, default_output, kind="output") for device in deduped]

    def _select_input_devices(self, devices: list[AudioDevice], default_input: int | None) -> list[AudioDevice]:
        filtered = [device for device in devices if self._is_meaningful_input_device(device)]
        deduped = self._dedupe_devices(filtered, kind="input")
        return [self._decorate_device(device, default_input, kind="input") for device in deduped]

    def _dedupe_devices(self, devices: list[AudioDevice], kind: str) -> list[AudioDevice]:
        grouped: dict[str, list[AudioDevice]] = {}
        for device in devices:
            key = self._device_group_key(device.name)
            grouped.setdefault(key, []).append(device)

        priority = self._OUTPUT_HOST_PRIORITY if kind == "output" else self._MIC_HOST_PRIORITY
        selected = [
            min(group, key=lambda item: (priority.get(item.hostapi or "", 99), int(item.id)))
            for group in grouped.values()
        ]
        return sorted(selected, key=lambda item: (not item.is_default, item.name.lower()))

    def _decorate_device(self, device: AudioDevice, default_id: int | None, kind: str) -> AudioDevice:
        return device.model_copy(
            update={
                "is_default": int(device.id) == default_id,
                "description": self._describe_device(device, kind),
            }
        )

    @staticmethod
    def _is_meaningful_output_device(device: AudioDevice) -> bool:
        name = device.name.strip()
        if not name:
            return False
        excluded_prefixes = (
            "Microsoft Sound Mapper",
            "Primary Sound Driver",
        )
        if name.startswith(excluded_prefixes):
            return False
        if "@System32" in name:
            return False
        if name.endswith("()"):
            return False
        return True

    @staticmethod
    def _is_meaningful_input_device(device: AudioDevice) -> bool:
        name = device.name.strip()
        if not name:
            return False
        excluded_prefixes = (
            "Microsoft Sound Mapper",
            "Primary Sound Capture Driver",
            "Primary Sound Driver",
        )
        if name.startswith(excluded_prefixes):
            return False
        if "@System32" in name:
            return False
        if name.endswith("()"):
            return False
        return True

    @staticmethod
    def _device_group_key(name: str) -> str:
        normalized = name.lower()
        normalized = re.sub(r"\b\d+\b", "", normalized)
        normalized = normalized.replace("with sst", "")
        normalized = normalized.replace("sg wave", "waves soundgrid")
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = normalized.replace("()", "")
        return normalized.strip()

    @staticmethod
    def _describe_device(device: AudioDevice, kind: str) -> str:
        lower_name = device.name.lower()
        if kind == "output":
            if "2nd output" in lower_name or "headphone" in lower_name:
                return "Wired headphones / headset output."
            if "speaker" in lower_name:
                return "Laptop speakers or monitor speakers."
            if "display audio" in lower_name or "hd audio driver for display audio" in lower_name:
                return "Monitor or HDMI audio output."
            return "Playback device for system audio capture."

        if "microphone array" in lower_name:
            return "Built-in laptop microphone array."
        if "headset" in lower_name:
            return "Headset microphone. Choose this for wired headphones with a mic."
        if "headphone" in lower_name and "mic" in lower_name:
            return "Headphone or headset microphone input."
        if "usb" in lower_name and ("mic" in lower_name or "microphone" in lower_name):
            return "External USB microphone or headset mic."
        if "line in" in lower_name:
            return "Line-in or external analog input. Do not use for normal speech unless intentional."
        if "stereo mix" in lower_name:
            return "System-mix style input. Not recommended for normal microphone use."
        if "mic" in lower_name or "microphone" in lower_name:
            return "Microphone input device."
        return "Audio input device."
