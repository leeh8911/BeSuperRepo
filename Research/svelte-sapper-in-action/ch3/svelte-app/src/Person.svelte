<script>
    export let age;
    export let ball;
    export let developer;
    export let favoriteColors;
    export let fullName;
    export let onBirthdat;
</script>

<div>
    {fullName} is {age} years old and {developer ? 'is' : 'is not'} a developer.
</div>
<div>
    They like the colors {favoriteColors.join(' and ')}.
</div>
<div>
    They like to throw {ball.new ? "a new" : "an old"} {ball.name} that weights {ball.grams} grams.
</div>
<button on:click={onBirthday}>It's my birthday!</button>



<style>


</style>