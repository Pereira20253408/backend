from notifications import enviar_alerta_telegram

mensaje = "Finanza Terminal Conectada! Tu bot de vigilancia esta activo."
resultado = enviar_alerta_telegram(mensaje)

if resultado:
    print("✅ Mensaje de prueba enviado con éxito.")
else:
    print("❌ Error al enviar el mensaje. Revisa el Token y el Chat ID.")
