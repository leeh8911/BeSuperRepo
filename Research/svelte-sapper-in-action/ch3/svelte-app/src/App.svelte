<script>
	export let name;
</script>

<main>
	<!-- <Person
	fullName="Jane Programmer"
	developer={true}
	ball={{name: 'baseball', grams: 149, new: false}}
	favoriteColors={['yellow', 'orange']}
	age={calculateAge(person)}
	onBirthday={celebrate}
	/> -->
	<h1>Hello {name}!</h1>
	<p>Visit the <a href="https://svelte.dev/tutorial">Svelte tutorial</a> to learn how to build Svelte apps.</p>
	<slot> This is devlog blog </slot>
	<p> This is devlog blog </p>
</main>

<style>
	main {
		text-align: center;
		padding: 1em;
		max-width: 240px;
		margin: 0 auto;
	}

	h1 {
		color: #ff3e00;
		text-transform: uppercase;
		font-size: 4em;
		font-weight: 100;
	}

	@media (min-width: 640px) {
		main {
			max-width: none;
		}
	}
</style>