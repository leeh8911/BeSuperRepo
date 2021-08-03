<script>
	import Clock from './Clock.svelte'
	let color = "red";
	let name = 'world';
	let upper = false;
	
	$: greeting = `Hello ${name}!`;
</script>

<label for="name">Name</label>
<input id="name" bind:value={name}>

<label for="color">Color</label>
<input id="color" type="color" bind:value={color}>
<div style="background-color: {color}" class="swatch"/>

<h1 style="color: {color}">{greeting}</h1>

<style>
	.swatch {
		display: inline-block;
		height: 20px;
		width: 20px;
	}
</style>
<Clock {color}/>