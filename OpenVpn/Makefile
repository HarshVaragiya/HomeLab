encrypt:
	tar -cz * | gpg -o backup.tar.gz.gpg --symmetric --cipher-algo AES256

decrypt:
	gpg -d backup.tar.gz.gpg | tar -xz
