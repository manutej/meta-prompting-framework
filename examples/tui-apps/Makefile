APPS := progress-timer file-browser system-monitor

.PHONY: build test vet smoke clean $(APPS)

build: $(APPS)

$(APPS):
	cd $@ && go build -o $@ .

vet:
	@for a in $(APPS); do (cd $$a && go vet ./...) || exit 1; done

test: vet
	@for a in $(APPS); do echo "== $$a" && (cd $$a && go test ./...) || exit 1; done

smoke: build
	python3 scripts/pty_smoke.py

clean:
	@for a in $(APPS); do rm -f $$a/$$a; done
