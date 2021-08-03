<script>
	export let color = 'blue';
	let hhmmss = '';
	setInterval(() => {
		hhmmss = new Date().toLocaleTimeString();
	}, 1000);
</script>

<span style="color: {color}">{hhmmss}</span>