"""Where the IC-705 is on the network, and the user it lets in: asked when the WiFi
radio is chosen from the SDR menu, and kept for next time (the password in the Keychain)."""

from __future__ import annotations

from PyQt6 import QtWidgets

from ..device.icom_net import NetworkLogin

SETUP_HELP = (
    "On the radio: SET > WLAN Set > WLAN ON, Connection Type “Station” to join "
    "your network (or “Access Point” and join the radio's own network from the "
    "Mac), then SET > WLAN Set > Remote Settings: Network Control ON, and a Network User1 "
    "ID and Password. The radio's address is under SET > WLAN Set > Connection Settings "
    "(Station) > IP Address (or the Mac's router lists it). Ports 50001-50003 as the radio "
    "sets them."
)


class NetworkLoginDialog(QtWidgets.QDialog):
    def __init__(self, login: NetworkLogin, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("IC-705 over WiFi")
        self.host = QtWidgets.QLineEdit(login.host)
        self.host.setPlaceholderText("192.168.1.50 or the radio's name")
        self.user = QtWidgets.QLineEdit(login.user)
        self.user.setPlaceholderText("Network User1 ID")
        self.password = QtWidgets.QLineEdit(login.password)
        self.password.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.password.setPlaceholderText("Network User1 Password")
        form = QtWidgets.QFormLayout()
        form.addRow("Radio address", self.host)
        form.addRow("User", self.user)
        form.addRow("Password", self.password)
        help_label = QtWidgets.QLabel(SETUP_HELP)
        help_label.setWordWrap(True)
        help_label.setMinimumWidth(420)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Connect")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(help_label)
        layout.addWidget(buttons)

    def login(self) -> NetworkLogin:
        return NetworkLogin(self.host.text().strip(), self.user.text().strip(),
                            self.password.text())

    def _accept(self) -> None:
        if not self.login().complete:
            QtWidgets.QMessageBox.warning(self, "IC-705 over WiFi",
                                          "The radio's address and user are both needed.")
            return
        self.accept()
